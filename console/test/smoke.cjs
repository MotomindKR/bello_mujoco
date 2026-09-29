const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const grpc = require("@grpc/grpc-js");
const loader = require("@grpc/proto-loader");
const { _electron: electron } = require("playwright");

const root = path.resolve(__dirname, "../..");
const capture = (page, options) =>
  page.screenshot({ timeout: process.env.CI ? 120000 : 30000, ...options });
async function closeApp(app) {
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    app.process().kill("SIGKILL");
  }, 10000);
  try {
    await app.close();
    assert(!timedOut, "Electron did not exit within 10 seconds");
  } finally {
    clearTimeout(timer);
  }
}
const load = (source) =>
  grpc.loadPackageDefinition(
    loader.loadSync(path.join(root, source), {
      longs: String,
      enums: String,
      defaults: true,
    }),
  );
const hardware = load(
  "deps/bello-stack/components/bello-hardware-api/proto/bello/hardware/v1/hardware.proto",
).bello.hardware.v1.HardwareService;
const pose = load(
  "deps/bello-stack/components/bello-pose-api/proto/bello/pose/v1/pose.proto",
).bello.pose.v1.PoseService;
const operator = load("rust/bello-policy/proto/operator.proto").bello.operator
  .v1.OperatorService;
const trackingApi = load("deps/pico-smplx-bridge/pose_stream.proto").pico.smplx
  .v1.PoseStream;
const { bodyNames } = require("../tracking.cjs");
let trackingAge = 0;
const trackingCalls = [];
const calls = [];
const samples = [];
const replacementSamples = [];
const servers = [];
const poseStreams = new Set();
let elbowPosition = 0;
let motorTemperature = 31;
let motorAge = 0;
let recoveryFails = false;
let runnerBusy = true;
let operatorSessions = 0;
const runnerStatus = {
  mode: "MODE_STAND",
  selected: "Motion",
  selectedMotion: "Wave",
  phase: "standing",
  modes: ["Stand", "Locomotion", "Motion", "Tracking"],
  startups: ["Crane", "Front-down"],
  selectedStartup: "Crane",
  sonicReady: true,
  sonicBufferedMs: 1100,
  sonicDelayMs: 1050,
  sonicStatus: "Ready",
  motions: ["Wave", "Bow"],
  controllerConnected: false,
  neutralRequired: true,
};
const powerSamples = [
  {
    sequence: 1,
    sampleAgeMs: "1000",
    motorRail: { voltageV: 48.125, currentA: 0.20000000298023224 },
  },
];
function repeating(call, value) {
  const timer = setInterval(() => call.write(value()), 40);
  call.on("cancelled", () => clearInterval(timer));
  call.on("error", () => clearInterval(timer));
  call.on("close", () => clearInterval(timer));
}
async function auditContrast(page) {
  const tokens = await page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    return Object.fromEntries(
      [
        "foreground",
        "body",
        "caption",
        "surface",
        "success",
        "success-background",
        "error",
        "error-background",
        "button-foreground",
        "button-primary-background",
      ].map((name) => [name, style.getPropertyValue("--" + name).trim()]),
    );
  });
  const luminance = (hex) => {
    let value = hex.replace("#", "");
    if (value.length === 3) value = [...value].map((c) => c + c).join("");
    const [r, g, b] = value
      .match(/../g)
      .map((v) => parseInt(v, 16) / 255)
      .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  for (const [text, background] of [
    ["foreground", "surface"],
    ["body", "surface"],
    ["caption", "surface"],
    ["success", "success-background"],
    ["error", "error-background"],
    ["button-foreground", "button-primary-background"],
  ]) {
    const a = luminance(tokens[text]),
      b = luminance(tokens[background]);
    assert(
      (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) >= 4.5,
      `${text} must meet AA contrast`,
    );
  }
}

async function serve(service, implementation) {
  const server = new grpc.Server();
  server.addService(service.service, implementation);
  const port = await new Promise((resolve, reject) =>
    server.bindAsync(
      "127.0.0.1:0",
      grpc.ServerCredentials.createInsecure(),
      (error, port) => (error ? reject(error) : resolve(port)),
    ),
  );
  servers.push(server);
  return `127.0.0.1:${port}`;
}
(async () => {
  const hardwareEndpoint = await serve(hardware, {
    ListHardware(_call, done) {
      calls.push("ListHardware");
      done(null, {
        motors: [{ name: "left_elbow", coordinate: "left_elbow_pitch_joint" }],
        groups: [],
        powerBoards: [
          { name: "main", staleTimeoutMs: "3000", syncPeriodMs: "1000" },
        ],
      });
    },
    WatchTelemetry(call) {
      calls.push("WatchTelemetry");
      for (const payload of [
        {
          health: {
            component: "can0",
            deviceStatus: { state: "DEVICE_STATE_ONLINE" },
            droppedEvents: "2",
          },
        },
        {
          group: {
            group: "body",
            state: "GROUP_STATE_RUNNING",
            controlTick: "123456789",
            jitterUs: "-12",
          },
        },
        {
          imu: {
            imuName: "base_imu",
            orientation: { w: 1 },
            angularVelocity: { x: 0.125 },
            sampleAgeMs: "0",
          },
        },
        {
          powerBoard: {
            boardName: "main",
            connectionGeneration: "1",
            status: { state: "DEVICE_STATE_ONLINE" },
            samples: powerSamples,
          },
        },
        {
          commandPolicy: {
            group: "body",
            motorName: "left_elbow",
            detail: "Diagnostic test event",
          },
        },
      ])
        repeating(call, () => payload);
      repeating(call, () => ({
        motor: {
          motorName: "left_elbow",
          position: elbowPosition,
          temperature: motorTemperature,
          feedbackAgeMs: motorAge,
        },
      }));
    },
  });
  const estimatorEndpoint = await serve(pose, {
    WatchPose(call) {
      poseStreams.add(call);
      repeating(call, () => ({
        referenceFrame: "odom",
        trackedFrame: "pelvis_link",
        sequence: "1",
        translationMetres: { z: 0.859 },
        orientationXyzw: { w: 1 },
        sampleAgeMs: "0",
      }));
    },
  });
  const runnerEndpoint = await serve(operator, {
    Exchange(call) {
      operatorSessions += 1;
      if (runnerBusy) {
        call.emit("error", {
          code: grpc.status.RESOURCE_EXHAUSTED,
          details:
            "Runner is in use by another console. Close that console to release its operator lease.",
        });
        return;
      }
      call.on("data", (sample) => {
        samples.push(sample);
        if (
          sample.selection &&
          runnerStatus.mode === "MODE_DISARMED" &&
          runnerStatus.startups.includes(sample.selection.startup)
        )
          runnerStatus.selectedStartup = sample.selection.startup;
        if (sample.selection && runnerStatus.mode === "MODE_STAND") {
          if (runnerStatus.modes.includes(sample.selection.mode)) {
            runnerStatus.selected = sample.selection.mode;
            runnerStatus.motionMenuOpen = sample.selection.mode === "Motion";
          }
          if (
            runnerStatus.selected === "Motion" &&
            runnerStatus.motions.includes(sample.selection.motion)
          ) {
            runnerStatus.selectedMotion = sample.selection.motion;
            runnerStatus.motionMenuOpen = false;
          }
          if (
            runnerStatus.selected === "Motion" &&
            sample.selection.motionMenuOpen != null
          )
            runnerStatus.motionMenuOpen = sample.selection.motionMenuOpen;
        }
        if (sample.clearFault && runnerStatus.canClearFault) {
          Object.assign(runnerStatus, {
            clearFaultSequence: sample.sequence,
            canClearFault: false,
            phase: "clearing fault",
            recoveryError: "",
          });
          setTimeout(
            () =>
              Object.assign(
                runnerStatus,
                recoveryFails
                  ? {
                      canClearFault: true,
                      phase: "fault",
                      recoveryError: "Feedback still stale.",
                    }
                  : {
                      mode: "MODE_DISARMED",
                      phase: "disarmed",
                      fault: "",
                      neutralRequired: true,
                    },
              ),
            150,
          );
        }
      });
      repeating(call, () => runnerStatus);
    },
  });
  const replacementRunner = await serve(operator, {
    Exchange(call) {
      call.on("data", (sample) => replacementSamples.push(sample));
      repeating(call, () => ({
        mode: "MODE_DISARMED",
        phase: "new robot",
        neutralRequired: true,
      }));
    },
  });
  const trackingEndpoint = await serve(trackingApi, {
    StreamPoses(call) {
      trackingCalls.push(call.request);
      repeating(call, () => ({
        sequence: "1",
        deviceId: "Synthetic headset",
        upstreamConnected: true,
        packetAgeSeconds: trackingAge,
        xrobotBody: {
          frame: "pico_native_y_up",
          valid: true,
          ageSeconds: trackingAge,
          jointNames: bodyNames,
          positions: bodyNames.flatMap((_, i) => [
            Math.sin(i) * 0.2,
            1 + i * 0.03,
            0,
          ]),
          orientations: bodyNames.flatMap(() => [1, 0, 0, 0]),
        },
      }));
    },
  });
  const configPath = path.join(root, "console/out/smoke-config.json");
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(
    configPath,
    JSON.stringify({
      ...require("../config.json"),
      hardware: hardwareEndpoint,
      estimator: estimatorEndpoint,
      runner: runnerEndpoint,
      tracking: trackingEndpoint,
    }),
  );
  let app;
  try {
    const launchOptions = {
      executablePath:
        process.env.BELLO_CONSOLE_EXECUTABLE || require("electron"),
      args: [
        ...(process.env.BELLO_CONSOLE_EXECUTABLE
          ? []
          : [path.join(root, "console")]),
        ...(process.env.CI && process.platform === "linux"
          ? ["--no-sandbox"]
          : []),
        ...(process.env.CI
          ? [
              "--use-gl=angle",
              "--use-angle=swiftshader",
              "--enable-unsafe-swiftshader",
              "--force-device-scale-factor=1",
            ]
          : []),
      ],
      env: { ...process.env, BELLO_CONSOLE_CONFIG: configPath },
    };
    app = await electron.launch(launchOptions);
    const page = await app.firstWindow();
    page.setDefaultTimeout(30000);
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error") console.error(message.text());
    });
    page.on("pageerror", (error) => errors.push(String(error)));
    await page
      .locator(".service")
      .filter({ hasText: "Runner" })
      .getByText("In use", { exact: true })
      .waitFor();
    assert.equal(
      samples.length,
      0,
      "A rejected console cannot send commands to the runner",
    );
    runnerBusy = false;
    await page.waitForFunction(
      () =>
        document.querySelector(".header-status strong")?.textContent ===
        "Stand",
    );
    await page.waitForFunction(
      () => document.querySelectorAll(".topbar .service.online").length === 4,
    );
    await page
      .waitForSelector('.model[data-ready="true"]', { timeout: 15000 })
      .catch(async (error) => {
        console.error(await page.locator(".model").getAttribute("data-error"));
        throw error;
      });
    {
      const saved = structuredClone(runnerStatus);
      Object.assign(runnerStatus, { mode: "MODE_DISARMED", phase: "disarmed" });
      await page
        .getByRole("button", { name: "Front-down", exact: true })
        .click();
      await page.waitForFunction(
        () =>
          document.querySelector('.mode-option[aria-pressed="true"]')
            ?.textContent === "Front-down",
      );
      assert(
        samples.some((sample) => sample.selection?.startup === "Front-down"),
      );
      assert.equal(runnerStatus.mode, "MODE_DISARMED");
      await capture(page, {
        path: path.join(root, "console/out/front-down-startup.png"),
      });
      Object.assign(runnerStatus, saved);
      await page.getByRole("button", { name: "Motion", exact: true }).waitFor();
    }
    const verifyStableOperatorCard = async () => {
      const original = structuredClone(runnerStatus);
      const bounds = () =>
        page
          .locator(".operator-card, .stick-dial, .abort-button")
          .evaluateAll((elements) =>
            elements.map((element) => {
              const r = element.getBoundingClientRect();
              return [r.x, r.y, r.width, r.height];
            }),
          );
      const baseline = await bounds();
      try {
        for (const update of [
          { mode: "MODE_DISARMED", phase: "disarmed" },
          { mode: "MODE_HOLD", phase: "ramping", progress: 0.25 },
          { mode: "MODE_HOLD", phase: "holding", progress: 1 },
          {
            mode: "MODE_HOLD",
            selectedStartup: "Front-down",
            phase: "bending",
            progress: 0.2,
          },
          { mode: "MODE_HOLD", phase: "awaiting lift", startupReady: false },
          { mode: "MODE_HOLD", phase: "awaiting lift", startupReady: true },
          { mode: "MODE_HOLD", phase: "straightening", progress: 0.5 },
          { mode: "MODE_HOLD", phase: "upright", progress: 1 },
          { mode: "MODE_HOLD", phase: "engaging ankles", progress: 0.5 },
          { mode: "MODE_LOCOMOTION", phase: "locomotion" },
          {
            mode: "MODE_TRACKING",
            phase: "motion",
            selected: "Tracking",
            sonicReady: false,
            sonicStatus:
              "Reference disconnected. Waiting for a fresh full-body reference.",
          },
          {
            mode: "MODE_MOTION",
            phase: "returning",
            activeMotion: "Long named motion for geometry verification",
            progress: 0.7,
          },
          {
            mode: "MODE_FAULT",
            phase: "fault",
            fault: "Long fault detail. ".repeat(80),
            canClearFault: true,
          },
        ]) {
          Object.assign(runnerStatus, update);
          await page.waitForFunction(
            (mode) =>
              document.querySelector(".header-status strong")?.textContent ===
              {
                MODE_DISARMED: "Disarmed",
                MODE_HOLD: "Hold",
                MODE_LOCOMOTION: "Locomotion",
                MODE_TRACKING: "Tracking",
                MODE_MOTION: "Motion",
                MODE_FAULT: "Fault",
              }[mode],
            update.mode,
          );
          assert.deepEqual(
            await bounds(),
            baseline,
            `Controller card and stick geometry must stay fixed in ${update.mode}`,
          );
          if (update.mode === "MODE_FAULT") {
            assert(
              await page.locator(".fault").evaluate((element) => {
                const region = element.getBoundingClientRect();
                return [
                  ".fault-recovery",
                  "#fault-clear-hint",
                  ".clear-fault",
                ].every((selector) => {
                  const target = element.querySelector(selector);
                  const rect = target.getBoundingClientRect();
                  return (
                    rect.top >= region.top &&
                    rect.bottom <= region.bottom &&
                    rect.left >= region.left &&
                    rect.right <= region.right &&
                    target.scrollHeight <= target.clientHeight
                  );
                });
              }),
              "Fault recovery controls and hint must remain fully visible with long details",
            );
            await capture(page, {
              fullPage: true,
              path: path.join(
                root,
                `console/out/console-fault-${await page.evaluate(() => innerWidth)}.png`,
              ),
            });
          }
        }
      } finally {
        for (const key of Object.keys(runnerStatus)) delete runnerStatus[key];
        Object.assign(runnerStatus, original);
        await page.waitForTimeout(120);
      }
    };
    const originalViewport = await page.evaluate(() => ({
      width: innerWidth,
      height: innerHeight,
    }));
    for (const [width, height] of [
      [1280, 720],
      [1366, 768],
      [1920, 1080],
      [2560, 1440],
    ]) {
      await page.setViewportSize({ width, height });
      await capture(page, {
        path: path.join(root, `console/out/overview-${width}x${height}.png`),
      });
      assert(
        await page.evaluate(
          () =>
            document.documentElement.scrollHeight <= innerHeight &&
            document.documentElement.scrollWidth <= innerWidth,
        ),
        `Overview must fit ${width}x${height} without page scrolling`,
      );
      assert(
        await page
          .locator(".stick-velocity")
          .evaluateAll((elements) =>
            elements.every(
              (element) =>
                element.scrollHeight <= element.clientHeight &&
                element.scrollWidth <= element.clientWidth,
            ),
          ),
        "Stick values must fit without internal scrolling",
      );
      await verifyStableOperatorCard();
    }
    await page.setViewportSize(originalViewport);
    const temperature = page.getByRole("status", { name: "Motor temperature" });
    await page.waitForFunction(() =>
      document
        .querySelector(".header-temperature")
        .textContent.includes("Temp OK"),
    );
    assert.equal(
      await temperature.evaluate((el) => el.classList.contains("text-success")),
      true,
    );
    const tempBounds = await temperature.boundingBox();
    motorTemperature = 60;
    await page.waitForFunction(
      () =>
        document.querySelector(".header-temperature").textContent === "Temp OK",
    );
    assert(
      await temperature.evaluate((el) => {
        const box = el.getBoundingClientRect(),
          label = el.querySelector("span").getBoundingClientRect();
        return (
          Math.abs((box.top + box.bottom - label.top - label.bottom) / 2) < 1
        );
      }),
      "Nominal temperature label must be vertically centered",
    );
    motorTemperature = 61.25;
    await page.waitForFunction(() =>
      document
        .querySelector(".header-temperature")
        .textContent.includes("61.25"),
    );
    assert.match(await temperature.innerText(), /left_elbow/);
    assert.equal(
      await temperature.evaluate((el) => el.classList.contains("text-error")),
      true,
    );
    assert.deepEqual(await temperature.boundingBox(), tempBounds);
    motorAge = 10000;
    await page.waitForFunction(() =>
      document
        .querySelector(".header-temperature")
        .textContent.includes("Temp --"),
    );
    assert.deepEqual(await temperature.boundingBox(), tempBounds);
    motorTemperature = 31;
    motorAge = 0;
    assert.deepEqual(
      await page.locator(".mode-option > span").allTextContents(),
      ["Stand", "Locomotion", "Motion", "Tracking"],
    );
    assert.equal(await page.locator(".mode-option em").count(), 0);
    const modeWidths = await page
      .locator(".mode-option")
      .evaluateAll((elements) =>
        elements.map((el) => el.getBoundingClientRect().width),
      );
    assert(
      Math.max(...modeWidths) - Math.min(...modeWidths) < 1,
      "Mode buttons share equal widths",
    );
    assert.equal(
      await page
        .locator(
          '.mode-row > .direction-hint[aria-label="D-pad left"] + .mode-list + .direction-hint[aria-label="D-pad right"]',
        )
        .count(),
      1,
    );
    await page.getByRole("button", { name: "Locomotion", exact: true }).click();
    await page.waitForFunction(
      () =>
        document.querySelector('.mode-option[aria-pressed="true"]')
          ?.textContent === "Locomotion",
    );
    await page.getByRole("switch", { name: "Upper-body teleop" }).waitFor();
    assert.equal(await page.locator(".motion-picker").count(), 0);
    await page.getByRole("button", { name: "Motion", exact: true }).click();
    await page.locator(".motion-choices").waitFor();
    assert.deepEqual(
      await page.locator(".motion-choice-name").allTextContents(),
      ["Wave", "Bow"],
    );
    assert.equal(
      await page
        .locator('.motion-choices li[aria-current="true"] .motion-choice-name')
        .textContent(),
      "Wave",
    );
    assert.equal(
      await page.locator(".motion-select-hint .gamepad-button").count(),
      1,
    );
    await page
      .locator(".motion-choices")
      .getByRole("button", { name: "Bow", exact: true })
      .click();
    await page.waitForFunction(() =>
      document
        .querySelector(".motion-picker-trigger")
        .textContent.includes("Bow"),
    );
    await page.locator(".motion-choices").waitFor({ state: "detached" });

    assert.deepEqual(
      samples
        .filter((sample) => sample.selection)
        .map((sample) =>
          sample.selection.startup != null
            ? { startup: sample.selection.startup }
            : sample.selection.mode != null
              ? { mode: sample.selection.mode }
              : { motion: sample.selection.motion },
        ),
      [
        { startup: "Front-down" },
        { mode: "Locomotion" },
        { mode: "Motion" },
        { motion: "Bow" },
      ],
    );
    await page.locator(".motion-picker-trigger").click();
    await page.locator(".motion-choices").waitFor();
    await page.locator(".motion-picker-trigger").click();
    await page.locator(".motion-choices").waitFor({ state: "detached" });
    assert.equal(
      samples.filter((sample) => sample.selection?.motionMenuOpen === true)
        .length,
      1,
    );
    await page.getByRole("button", { name: "Stand", exact: true }).click();
    await page.waitForFunction(
      () =>
        document.querySelector('.mode-option[aria-pressed="true"]')
          ?.textContent === "Stand",
    );
    assert.equal(
      await page.locator(".motion-picker, .tracking-toggle").count(),
      0,
    );
    assert.match(
      (await page.evaluate(() => window.bello.selectChoice("mode", "Unknown")))
        .error,
      /unavailable/,
    );
    assert.match(
      (await page.evaluate(() => window.bello.selectChoice("abort", "Motion")))
        .error,
      /Invalid/,
    );
    runnerStatus.mode = "MODE_LOCOMOTION";
    await page.waitForFunction(
      () => document.querySelector(".mode-option")?.disabled,
    );
    assert.match(
      (await page.evaluate(() => window.bello.selectChoice("mode", "Motion")))
        .error,
      /Stand/,
    );
    runnerStatus.mode = "MODE_STAND";
    await page.waitForFunction(
      () => !document.querySelector(".mode-option")?.disabled,
    );

    await app.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows()[0].focus(),
    );
    await page.waitForFunction(() => document.hasFocus());
    runnerStatus.abortProgress = 0.5;
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".abort-chord .gamepad-hold-fill")].every(
        (el) => el.style.transform === "scaleX(0.5)",
      ),
    );
    runnerStatus.abortProgress = 0;
    await page.waitForFunction(
      () =>
        document.querySelector(".abort-chord .gamepad-hold-fill").style
          .transform === "scaleX(0)",
    );
    const abortButton = page.getByRole("button", {
      name: "Hold to abort",
      exact: true,
    });
    await abortButton.hover();
    await page.mouse.down();
    await page.waitForTimeout(300);
    await page.mouse.up();
    await page.waitForTimeout(800);
    assert.equal(
      samples.filter((sample) => sample.abort).length,
      0,
      "Short press must not abort",
    );
    await abortButton.hover();
    await page.mouse.down();
    await page.waitForTimeout(1200);
    if (!samples.some((sample) => sample.abort)) {
      console.error(
        await page.evaluate(() => ({
          focused: document.hasFocus(),
          holding: document.querySelector(".abort-button").className,
          label: document.querySelector(".abort-button").textContent,
        })),
      );
      await capture(page, {
        path: path.join(root, "console/out/console-abort-failure.png"),
      });
    }
    await page.mouse.up();
    assert.equal(
      samples.filter((sample) => sample.abort).length,
      1,
      "Full hold sends exactly one abort",
    );
    await abortButton.focus();
    await page.keyboard.down("Space");
    await page.waitForTimeout(300);
    await page.keyboard.up("Space");
    await page.waitForTimeout(800);
    assert.equal(
      samples.filter((sample) => sample.abort).length,
      1,
      "Short keyboard hold must cancel",
    );
    assert.equal(await page.locator("canvas").count(), 1);
    assert.equal(await page.locator(".model .model-feedback").count(), 1);
    assert.equal(await page.locator(".hero-heading").count(), 0);
    const cardBounds = await page.locator(".hero").boundingBox();
    assert.deepEqual(
      await page.locator(".hero .model").boundingBox(),
      cardBounds,
    );
    assert.equal(
      await page
        .getByRole("heading", { name: "Robot state", exact: true })
        .count(),
      0,
    );
    assert.equal(await page.locator(".model").getAttribute("data-error"), "");
    await capture(page, {
      path: path.join(root, "console/out/console-smoke.png"),
    });
    assert.equal(
      await page
        .getByRole("heading", { name: "Controller & status", exact: true })
        .count(),
      0,
    );
    assert.equal(await page.locator(".stick-dial").count(), 2);
    assert.equal(await page.locator(".operator-guide p").count(), 0);
    runnerStatus.mode = "MODE_LOCOMOTION";
    runnerStatus.velocityCommand = {
      forwardMps: 0.35,
      rightMps: -0.2,
      yawRps: 0.4,
    };
    await page
      .locator(".sticks")
      .getByText("0.35 m/s", { exact: true })
      .waitFor();
    await page
      .locator(".sticks")
      .getByText("-0.2 m/s", { exact: true })
      .waitFor();
    await page
      .locator(".sticks")
      .getByText("0.4 rad/s", { exact: true })
      .waitFor();
    await capture(page, {
      path: path.join(root, "console/out/console-sticks.png"),
    });
    runnerStatus.mode = "MODE_STAND";
    runnerStatus.velocityCommand = null;
    await page
      .locator(".sticks")
      .getByText("Inactive", { exact: true })
      .waitFor();
    assert.equal(
      await page
        .locator(".sticks")
        .getByText("0.35 m/s", { exact: true })
        .count(),
      0,
    );
    await auditContrast(page);
    const sessionsBeforeTheme = operatorSessions;
    for (const [theme, label, dark] of [
      ["light", "Light", false],
      ["dark", "Dark", true],
      ["system", "System", true],
    ]) {
      await page.getByRole("button", { name: "Settings", exact: true }).click();
      await page.getByRole("radio", { name: label, exact: true }).check();
      await page.getByRole("button", { name: "Save", exact: true }).click();
      await page.waitForFunction(
        (value) =>
          document.documentElement.classList.contains("dark") === value,
        dark,
      );
      assert.equal(JSON.parse(fs.readFileSync(configPath)).theme, theme);
      if (theme === "dark") {
        await page.emulateMedia({ colorScheme: "light" });
        assert.equal(
          await page.evaluate(() =>
            document.documentElement.classList.contains("dark"),
          ),
          true,
        );
        await page.emulateMedia({ colorScheme: "dark" });
      }
      if (theme === "system") {
        await page.emulateMedia({ colorScheme: "light" });
        await page.waitForFunction(
          () => !document.documentElement.classList.contains("dark"),
        );
        await page.emulateMedia({ colorScheme: "dark" });
        await page.waitForFunction(() =>
          document.documentElement.classList.contains("dark"),
        );
      }
    }
    assert.equal(
      operatorSessions,
      sessionsBeforeTheme,
      "Appearance changes must not reconnect control",
    );
    const power = page.locator(".header-power");
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Stand",
    );
    await power.getByText("48.125 V", { exact: false }).waitFor();
    await power.getByText("0.2 A", { exact: true }).waitFor();
    assert.equal(
      await power.locator(".power-stale").count(),
      0,
      "Power samples within the board's timeout remain fresh beyond 250 ms",
    );
    powerSamples[0].sampleAgeMs = "3100";
    await power.locator(".power-stale").waitFor();
    powerSamples[0].sampleAgeMs = "1000";
    powerSamples[0].stale = true;
    await page.waitForTimeout(200);
    assert.equal(
      await power.locator(".power-stale").count(),
      1,
      "Server stale flags take priority over local age",
    );
    powerSamples[0].stale = false;
    await power.locator(".power-stale").waitFor({ state: "detached" });
    assert(
      (await power.textContent()).includes("-- %"),
      "Absent charge must remain unknown",
    );
    assert.equal(
      Number(await power.locator(".battery-fill").getAttribute("width")),
      0,
    );
    await power
      .getByRole("img", { name: "Charging status unavailable", exact: true })
      .waitFor();
    powerSamples.push({
      sequence: 2,
      sampleAgeMs: "0",
      battery: { voltageV: 49.25, currentA: 0, socPercent: 82.5 },
    });
    await power.getByText("82.5 %", { exact: true }).waitFor();
    await power.locator(".battery-normal").waitFor();
    powerSamples[1].battery.socPercent = 20;
    await power.locator(".battery-low").waitFor();
    powerSamples[1].battery.socPercent = 10;
    await power.locator(".battery-critical").waitFor();
    powerSamples[1].stale = true;
    await power.locator(".battery-unknown").waitFor();
    powerSamples[1].stale = false;
    powerSamples[1].battery.socPercent = 82.5;
    await power.getByText("82.5 %", { exact: true }).waitFor();
    assert(
      Math.abs(
        Number(await power.locator(".battery-fill").getAttribute("width")) -
          9.9,
      ) < 1e-8,
    );
    powerSamples.push({
      sequence: 3,
      sampleAgeMs: "0",
      iwc: {
        info: { bits: 32, activeNames: ["BATTERY_CHARGING"] },
        critical: { bits: 4, activeNames: ["Motor rail overvoltage"] },
      },
    });
    await power.getByRole("img", { name: "Charging", exact: true }).waitFor();
    const chargingBounds = await power.boundingBox();
    powerSamples[2].stale = true;
    await power
      .getByRole("img", { name: "Charging status unavailable", exact: true })
      .waitFor();
    powerSamples[2].stale = false;
    powerSamples[2].iwc.info = { bits: 0, activeNames: [] };
    await power
      .getByRole("img", { name: "Not charging", exact: true })
      .waitFor();
    assert.deepEqual(
      await power.boundingBox(),
      chargingBounds,
      "Charging state must not shift power telemetry",
    );
    await page
      .locator(".fault.active")
      .filter({ hasText: "Motor rail overvoltage" })
      .waitFor();
    powerSamples.pop();
    await page.waitForFunction(() => !document.querySelector(".fault.active"));
    const headerBoxes = () =>
      page
        .locator(
          ".header-status, .header-indicators, .header-power, .header-power-metric, .connections, .service",
        )
        .evaluateAll((elements) =>
          elements.map((element) => {
            const rect = element.getBoundingClientRect();
            return [rect.x, rect.y, rect.width, rect.height];
          }),
        );
    const beforeDigits = await headerBoxes();
    const railTextEdges = await page
      .locator(".power-rail")
      .evaluate((element) =>
        [".power-metric-heading > span", ".power-reading > strong"].map(
          (selector) => {
            const range = document.createRange();
            range.selectNodeContents(element.querySelector(selector));
            return range.getBoundingClientRect().x;
          },
        ),
      );
    assert.equal(
      railTextEdges[0],
      railTextEdges[1],
      "Motor rail label and voltage must share their left edge",
    );
    powerSamples[0].motorRail = { voltageV: 8, currentA: -123.125 };
    powerSamples[1].battery.socPercent = 1;
    Object.assign(runnerStatus, {
      mode: "MODE_FAULT",
      phase: "fault",
      fault: "CAN queue test fault",
      canClearFault: true,
    });
    await power.getByText("-123.125 A", { exact: true }).waitFor();
    assert.equal(
      Number(await power.locator(".battery-fill").getAttribute("width")),
      0.12,
    );
    await page
      .getByRole("button", { name: "Clear fault", exact: true })
      .waitFor();
    await capture(page, {
      fullPage: true,
      path: path.join(root, "console/out/console-fault.png"),
    });
    assert.deepEqual(
      await headerBoxes(),
      beforeDigits,
      "Top-bar geometry must not change with digit count or status",
    );
    recoveryFails = true;
    await page
      .getByRole("button", { name: "Clear fault", exact: true })
      .click();
    await page
      .getByRole("alert")
      .filter({ hasText: "Feedback still stale." })
      .waitFor();
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Fault",
    );
    recoveryFails = false;
    await page
      .getByRole("button", { name: "Clear fault", exact: true })
      .click();
    await page.waitForFunction(
      () =>
        document.querySelector(".header-status strong")?.textContent ===
        "Disarmed",
    );
    assert.equal(samples.filter((sample) => sample.clearFault).length, 2);
    Object.assign(runnerStatus, { mode: "MODE_STAND", phase: "standing" });
    powerSamples[0].motorRail = {
      voltageV: 48.125,
      currentA: 0.20000000298023224,
    };
    powerSamples[1].battery.socPercent = 82.5;
    assert.equal(
      await page.getByText("POLICY CONSOLE", { exact: true }).count(),
      0,
    );
    assert.equal(
      await page
        .getByRole("button", { name: "Settings", exact: true })
        .textContent(),
      "",
    );
    await page.getByRole("tab", { name: "Tracking", exact: true }).click();
    await page.locator(".tracking-canvas canvas").waitFor();
    await page.waitForFunction(() =>
      document
        .querySelector(".tracking-feedback strong")
        ?.textContent.startsWith("Live"),
    );
    const trackingBadge = page
      .locator(".topbar .service")
      .filter({ hasText: "Tracking" });
    await trackingBadge.getByText(/^Live/).waitFor();
    assert.equal(trackingCalls[0].includeRaw, false);
    assert.equal(trackingCalls[0].maxRateHz, 30);
    await page.getByRole("button", { name: "Center tracking pose" }).click();
    await capture(page, {
      path: path.join(root, "console/out/console-tracking.png"),
    });
    trackingAge = 2;
    await page.waitForFunction(() =>
      document
        .querySelector(".tracking-feedback strong")
        ?.textContent.startsWith("Stale"),
    );
    await trackingBadge.getByText("Stale", { exact: true }).waitFor();
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Stand",
    );
    trackingAge = 0;
    await page.getByRole("tab", { name: "Hardware", exact: true }).click();
    await page
      .getByRole("heading", { name: "Component health", exact: true })
      .waitFor();
    const rawCurrent = page.locator('td[title="0.20000000298023224"]');
    await rawCurrent.waitFor();
    assert.equal(await rawCurrent.textContent(), "0.2");
    assert(
      await page
        .locator(".diagnostic-grid")
        .textContent()
        .then(
          (text) =>
            text.includes("48.125") &&
            text.includes("123456789") &&
            text.includes("can0"),
        ),
    );
    await capture(page, {
      path: path.join(root, "console/out/console-hardware.png"),
    });
    await page.getByLabel("Search status").fill("no-matching-motor");
    assert.equal(await page.locator(".motor-table tbody tr").count(), 0);
    await page.getByLabel("Search status").fill("left_elbow");
    assert.equal(await page.locator(".motor-table tbody tr").count(), 1);
    await page.getByRole("tab", { name: "Pose", exact: true }).click();
    await page.getByRole("cell", { name: "0.859", exact: true }).waitFor();
    await page.getByRole("cell", { name: "0.125", exact: true }).waitFor();
    await page.getByRole("tab", { name: "Operator", exact: true }).click();
    assert.equal(
      await page
        .getByRole("button", { name: /Pause display|Resume display/ })
        .count(),
      0,
    );
    assert.equal(
      await page
        .locator(".diagnostic-toolbar h1, .diagnostic-toolbar p")
        .count(),
      0,
    );
    const samplePanel = page.locator(".diagnostic-card").filter({
      has: page.getByRole("heading", {
        name: "Local controller sample",
        exact: true,
      }),
    });
    const initialSample = await samplePanel.textContent();
    await page.waitForFunction(
      (oldText) =>
        [...document.querySelectorAll(".diagnostic-card")].find(
          (card) =>
            card.querySelector("h2")?.textContent === "Local controller sample",
        )?.textContent !== oldText,
      initialSample,
    );
    await page.getByRole("tab", { name: "Services", exact: true }).click();
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Stand",
    );
    assert(
      (await page.locator(".service-table").textContent()).includes(
        runnerEndpoint,
      ),
    );
    await page
      .getByRole("tab", { name: "Services", exact: true })
      .press("Home");
    assert.equal(
      await page
        .getByRole("tab", { name: "Overview", exact: true })
        .getAttribute("aria-selected"),
      "true",
    );
    assert.equal(await page.locator("canvas").count(), 1);
    const streamCalls = calls.length;
    const sessionsBeforeStyle = operatorSessions;
    let lastSequence = BigInt(samples.at(-1).sequence);
    for (const [label, value, shoulder] of [
      ["PlayStation", "playstation", "L1"],
      ["Xbox", "xbox", "LB"],
      ["Switch", "switch", "L"],
    ]) {
      await page.getByRole("button", { name: "Settings", exact: true }).click();
      await page.getByRole("radio", { name: label, exact: true }).check();
      assert.equal(
        await page
          .locator("dialog .gamepad-button img")
          .evaluateAll((images) =>
            images.every((img) => img.complete && img.naturalWidth === 128),
          ),
        true,
      );
      await page.getByRole("button", { name: "Save", exact: true }).click();
      await page.waitForFunction(
        (style) =>
          document.querySelector(".abort-chord kbd")?.dataset
            .controllerStyle === style,
        value,
      );
      assert.equal(
        await page
          .locator(".abort-chord kbd")
          .first()
          .getAttribute("aria-label"),
        shoulder,
      );
      assert.equal(
        JSON.parse(fs.readFileSync(configPath)).controllerStyle,
        value,
      );
      assert.equal(
        calls.length,
        streamCalls,
        "Changing prompt style must preserve service connections",
      );
      assert.equal(
        operatorSessions,
        sessionsBeforeStyle,
        "Changing prompt style must preserve the operator lease",
      );
      assert(
        BigInt(samples.at(-1).sequence) > lastSequence,
        "Operator sequence must remain continuous",
      );
      lastSequence = BigInt(samples.at(-1).sequence);
    }
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    assert.equal(
      await page
        .getByRole("radio", { name: "Switch", exact: true })
        .isChecked(),
      true,
    );
    await page.getByRole("radio", { name: "Xbox", exact: true }).check();
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(
      JSON.parse(fs.readFileSync(configPath)).controllerStyle,
      "switch",
    );
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    await page.getByLabel("Robot IP or hostname").fill("999.42.0.1");
    await page.getByRole("button", { name: "Save" }).click();
    await page
      .getByRole("alert")
      .filter({ hasText: "valid hardware host" })
      .waitFor();
    assert.equal(
      JSON.parse(fs.readFileSync(configPath)).hardware,
      hardwareEndpoint,
    );
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Stand",
    );
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    assert.equal(
      await page.getByLabel("Robot IP or hostname").inputValue(),
      "127.0.0.1",
    );
    await page.getByLabel("Robot IP or hostname").fill("localhost");
    await page
      .getByText("Individual service addresses", { exact: true })
      .click();
    assert.equal(
      await page
        .getByRole("textbox", { name: "Hardware", exact: true })
        .inputValue(),
      hardwareEndpoint.replace("127.0.0.1", "localhost"),
    );
    await capture(page, {
      path: path.join(root, "console/out/console-settings.png"),
    });
    await page.getByRole("button", { name: "Save" }).click();
    await page.waitForFunction(() => !document.querySelector("dialog"));
    await page.waitForFunction(
      () => document.querySelectorAll(".topbar .service.online").length === 4,
    );
    assert.equal(
      JSON.parse(fs.readFileSync(configPath)).runner,
      runnerEndpoint.replace("127.0.0.1", "localhost"),
    );
    elbowPosition = -1.2;
    await page.locator(".feedback-motors > summary").click();
    await page.waitForFunction(
      () =>
        document.querySelector(".motor-list strong")?.textContent ===
        "-1.2 rad",
    );
    await capture(page, {
      path: path.join(root, "console/out/console-bent.png"),
    });
    await page.locator(".feedback-motors > summary").click();
    await page.emulateMedia({ colorScheme: "dark" });
    await page.waitForFunction(
      () =>
        getComputedStyle(document.documentElement)
          .getPropertyValue("--background")
          .trim() === "#15100f",
    );
    await page.waitForSelector('.model[data-ready="true"]');
    await capture(page, {
      path: path.join(root, "console/out/console-dark.png"),
    });
    await auditContrast(page);
    await page.setViewportSize({ width: 375, height: 812 });
    await verifyStableOperatorCard();
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await capture(page, {
      path: path.join(root, "console/out/console-mobile.png"),
    });
    for (const name of [
      "Hardware",
      "Pose",
      "Operator",
      "Services",
      "Tracking",
    ]) {
      await page.getByRole("tab", { name, exact: true }).click();
      if (name === "Tracking") {
        trackingAge = 2;
        await page
          .locator(".tracking-toolbar .service")
          .getByText("Stale", { exact: true })
          .waitFor();
        assert(
          await page
            .locator(".tracking-toolbar .service")
            .evaluate((element) => {
              const box = element.getBoundingClientRect();
              return [...element.children].every((child) => {
                const rect = child.getBoundingClientRect();
                return (
                  rect.left >= box.left &&
                  rect.right <= box.right &&
                  rect.top >= box.top &&
                  rect.bottom <= box.bottom
                );
              });
            }),
          "Tracking badge must contain its label and stale status on mobile",
        );
        await capture(page, {
          path: path.join(
            root,
            "console/out/console-tracking-mobile-stale.png",
          ),
        });
        trackingAge = 0;
      }
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `${name} must fit mobile width`,
      );
    }
    await capture(page, {
      path: path.join(root, "console/out/console-services-mobile-dark.png"),
    });
    await page.getByRole("tab", { name: "Overview", exact: true }).click();
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    await page
      .getByText("Individual service addresses", { exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Runner", exact: true })
      .fill(replacementRunner);
    await capture(page, {
      path: path.join(root, "console/out/console-settings-mobile-dark.png"),
    });
    assert(
      await page
        .locator("dialog")
        .evaluate((element) => element.scrollWidth <= element.clientWidth),
    );
    await page.getByRole("button", { name: "Save" }).click();
    await page.waitForFunction(
      () =>
        document.querySelector(".header-status strong")?.textContent ===
        "Disarmed",
    );
    const oldSampleCount = samples.length;
    await page.waitForTimeout(1200);
    assert.equal(
      samples.length,
      oldSampleCount,
      "Previous runner must not reconnect or receive samples",
    );
    assert(replacementSamples.length > 3);
    for (const stream of poseStreams) stream.destroy();
    servers[1].forceShutdown();
    await page.waitForFunction(
      () => document.querySelectorAll(".topbar .service.online").length === 3,
    );
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Disarmed",
    );
    servers[0].forceShutdown();
    await page.waitForFunction(
      () => document.querySelectorAll(".topbar .service.online").length === 2,
    );
    await page.locator(".feedback-motors > summary").click();
    await page.waitForSelector(".model-feedback .motor-list .stale");
    await page.waitForSelector(".header-power .power-stale");
    const hardwareBadge = page
      .locator(".service")
      .filter({ has: page.locator("strong", { hasText: "Hardware" }) });
    await page.mouse.move(0, 0);
    assert.equal(await hardwareBadge.getByRole("tooltip").isVisible(), false);
    await hardwareBadge.hover();
    await hardwareBadge.getByRole("tooltip").waitFor({ state: "visible" });
    assert((await hardwareBadge.getByRole("tooltip").textContent()).length > 0);
    await page.mouse.move(0, 0);
    await hardwareBadge.focus();
    await hardwareBadge.getByRole("tooltip").waitFor({ state: "visible" });
    assert.equal(
      await page.locator(".header-status strong").textContent(),
      "Disarmed",
    );
    const native = await app.evaluate(({ app }) => {
      const load = process
        .getBuiltinModule("module")
        .createRequire(app.getAppPath() + "/main.cjs");
      const sdl = load("@kmamal/sdl");
      return {
        loaded: Boolean(sdl.controller),
        devices: sdl.controller.devices.length,
      };
    });
    assert.equal(native.loaded, true);
    assert(samples.length > 3);
    assert(
      samples.every(
        (value, index) =>
          index === 0 ||
          BigInt(value.sequence) > BigInt(samples[index - 1].sequence),
      ),
    );
    assert.deepEqual([...new Set(calls)].sort(), [
      "ListHardware",
      "WatchTelemetry",
    ]);
    assert.deepEqual(errors, []);
    await page
      .getByRole("button", { name: "Enter fullscreen", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Exit fullscreen", exact: true })
      .waitFor();
    assert(
      await app.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows()[0].isFullScreen(),
      ),
    );
    await page
      .getByRole("button", { name: "Exit fullscreen", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Enter fullscreen", exact: true })
      .waitFor();
    assert.equal(
      await app.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows()[0].isFullScreen(),
      ),
      false,
    );
    await closeApp(app);
    app = await electron.launch(launchOptions);
    const restarted = await app.firstWindow();
    restarted.setDefaultTimeout(30000);
    await restarted.waitForFunction(
      () =>
        document.querySelector(".header-status strong")?.textContent ===
        "Disarmed",
    );
    await restarted
      .getByRole("button", { name: "Settings", exact: true })
      .click();
    assert.equal(
      await restarted
        .getByRole("radio", { name: "Switch", exact: true })
        .isChecked(),
      true,
    );
    await restarted
      .getByText("Individual service addresses", { exact: true })
      .click();
    assert.equal(
      await restarted
        .getByRole("textbox", { name: "Runner", exact: true })
        .inputValue(),
      replacementRunner,
    );
    await restarted
      .getByRole("button", { name: "Cancel", exact: true })
      .click();
    assert.equal(
      await restarted.locator(".header-status strong").textContent(),
      "Disarmed",
    );
    servers[3].forceShutdown();
    await restarted.locator('.header-status[data-fresh="false"]').waitFor();
    assert.equal(
      await restarted.locator(".header-status strong").textContent(),
      "Disarmed",
    );
    const resizeState = await app.evaluate(
      async ({ BrowserWindow, screen }) => {
        const window = BrowserWindow.getAllWindows()[0];
        const original = window.getBounds();
        const area = screen.getDisplayMatching(original).workArea;
        const resizable = window.isResizable();
        const sizes = [];
        const requested = [
          [800, 480],
          [1000, 700],
        ].map(([width, height]) => [
          Math.min(width, area.width),
          Math.min(height, area.height),
        ]);
        window.setPosition(area.x, area.y);
        for (const [width, height] of requested) {
          window.setSize(width, height);
          await new Promise((resolve) => setTimeout(resolve, 100));
          sizes.push(window.getSize());
        }
        window.setBounds(original);
        return { resizable, sizes, requested };
      },
    );
    assert.equal(
      resizeState.resizable,
      true,
      "Native window must allow resizing",
    );
    assert.deepEqual(resizeState.sizes, resizeState.requested);
    assert.notDeepEqual(resizeState.sizes[0], resizeState.sizes[1]);
    console.log(
      JSON.stringify({
        result: "passed",
        samples: samples.length,
        native,
        screenshot: "console/out/console-smoke.png",
      }),
    );
  } catch (error) {
    console.error(error);
    throw error;
  } finally {
    try {
      if (app) await closeApp(app);
    } finally {
      for (const server of servers) server.forceShutdown();
    }
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
