const { app, BrowserWindow, protocol, net, ipcMain } = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const { pathToFileURL } = require("node:url");
const grpc = require("@grpc/grpc-js");
const loader = require("@grpc/proto-loader");
const { validateConnections, validateSettings } = require("./settings.cjs");
const { sampleController } = require("./controller.cjs");
const { trackingFrame } = require("./tracking.cjs");
const { emptyTelemetry, displaySnapshot } = require("./snapshot.cjs");

protocol.registerSchemesAsPrivileged([
  {
    scheme: "bello",
    privileges: {
      secure: true,
      standard: true,
      supportFetchAPI: true,
      corsEnabled: true,
    },
  },
]);

const root = path.resolve(__dirname, "..");
const resource = (name, source) =>
  app.isPackaged
    ? path.join(process.resourcesPath, name)
    : path.join(root, source);
const schema = (name, source) =>
  grpc.loadPackageDefinition(
    loader.loadSync(resource(name, source), {
      keepCase: false,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    }),
  );
const settingsPath =
  process.env.BELLO_CONSOLE_CONFIG ||
  path.join(app.getPath("userData"), "config.json");
const config = {
  ...JSON.parse(fs.readFileSync(path.join(__dirname, "config.json"), "utf8")),
  ...(fs.existsSync(settingsPath)
    ? JSON.parse(fs.readFileSync(settingsPath, "utf8"))
    : {}),
};
if (process.env.BELLO_CONSOLE_CONFIG && !fs.existsSync(settingsPath))
  throw new Error("Console configuration file does not exist");
Object.assign(config, validateSettings(config));
if (
  !Number.isInteger(config.controllerIndex) ||
  config.controllerIndex < 0 ||
  !Number.isFinite(config.deadzone) ||
  config.deadzone < 0 ||
  config.deadzone >= 1 ||
  !Number.isFinite(config.freshnessMs) ||
  config.freshnessMs <= 0 ||
  !config.referenceFrame ||
  !config.trackedFrame
)
  throw new Error("Invalid console configuration");

const snapshot = {
  endpoints: validateConnections(config),
  controllerStyle: config.controllerStyle,
  theme: config.theme,
  services: {
    hardware: { connected: false, last: 0, error: "" },
    estimator: { connected: false, last: 0, error: "" },
    runner: { connected: false, last: 0, error: "" },
    tracking: { connected: false, last: 0, error: "" },
  },
  controller: { connected: false, name: "", error: "" },
  ...emptyTelemetry(),
  controllerSample: null,
  lastSentSequence: null,
  displayConfig: {
    freshnessMs: config.freshnessMs,
    referenceFrame: config.referenceFrame,
    trackedFrame: config.trackedFrame,
    controllerIndex: config.controllerIndex,
    deadzone: config.deadzone,
  },
};
let window;
let controller = null;
let sdl;
let sequence = 0n;
let operatorCall;
let pendingClear;
let abortHoldTimer;
let pendingAbort = false;
let pendingSelection;
function cancelAbortHold() {
  clearTimeout(abortHoldTimer);
  abortHoldTimer = null;
}

function finishClear(error) {
  if (!pendingClear) return;
  clearTimeout(pendingClear.timer);
  pendingClear.resolve(error ? { error } : {});
  pendingClear = null;
}
const streams = new Set();
const clients = new Set();
const timers = new Set();
const retryTimers = new Set();
let connectionGeneration = 0;
let quitting = false;
function every(callback, milliseconds) {
  const timer = setInterval(callback, milliseconds);
  timers.add(timer);
  return timer;
}
function retryLater(callback) {
  if (!quitting) {
    const timer = setTimeout(() => {
      retryTimers.delete(timer);
      callback();
    }, 1000);
    retryTimers.add(timer);
  }
}

function mark(service, error = "") {
  const item = snapshot.services[service];
  item.connected = !error;
  item.error = error;
  if (!error) item.busy = false;
  if (!error) {
    item.last = Date.now();
    item.messages = (item.messages || 0) + 1;
  }
}
function startController() {
  try {
    sdl = require("@kmamal/sdl");
    const scan = () => {
      try {
        if (controller && !controller.closed) return;
        const device = sdl.controller.devices.filter((item) => item.mapping)[
          config.controllerIndex
        ];
        if (!device) {
          controller = null;
          snapshot.controller = {
            connected: false,
            name: "",
            error: "No mapped controller",
          };
          return;
        }
        const opened = sdl.controller.openDevice(device);
        controller = opened;
        snapshot.controller = {
          connected: true,
          name: device.name || "Controller",
          error: "",
        };
        opened.on("close", () => {
          if (controller !== opened) return;
          controller = null;
          snapshot.controller = {
            connected: false,
            name: "",
            error: "Controller disconnected",
          };
        });
      } catch (error) {
        controller = null;
        snapshot.controller = {
          connected: false,
          name: "",
          error: String(error),
        };
      }
    };
    scan();
    every(scan, 1000);
  } catch (error) {
    snapshot.controller.error = String(error);
  }
}
function controllerSample() {
  sequence += 1n;
  const sample = sampleController(controller, config.deadzone);
  return {
    sequence: sequence.toString(),
    sampleMonotonicNs: process.hrtime.bigint().toString(),
    ...sample,
  };
}
function reconnectingStream(service, create) {
  const generation = connectionGeneration;
  const current = () => !quitting && generation === connectionGeneration;
  const connect = () => {
    if (!current()) return;
    snapshot.services[service].attempts =
      (snapshot.services[service].attempts || 0) + 1;
    let call;
    try {
      call = create(current);
    } catch (error) {
      mark(service, String(error));
      retryLater(connect);
      return;
    }
    streams.add(call);
    let ended = false;
    const retry = (error) => {
      if (ended) return;
      ended = true;
      if (current()) {
        snapshot.services[service].busy =
          service === "runner" &&
          error?.code === grpc.status.RESOURCE_EXHAUSTED;
        mark(service, error?.details || String(error || "Stream ended"));
      }
      streams.delete(call);
      call.cancel();
      if (current()) retryLater(connect);
    };
    call.on("error", retry);
    call.on("end", () => retry("Stream ended"));
  };
  connect();
}
function startHardware() {
  const api = schema(
    "hardware.proto",
    "deps/bello-stack/components/bello-hardware-api/proto/bello/hardware/v1/hardware.proto",
  );
  const client = new api.bello.hardware.v1.HardwareService(
    config.hardware,
    grpc.credentials.createInsecure(),
  );
  clients.add(client);
  reconnectingStream("hardware", (current) => {
    client.ListHardware({}, (error, inventory) => {
      if (!current()) return;
      if (error) {
        mark("hardware", error.message);
        return;
      }
      snapshot.inventory = inventory;
      mark("hardware");
    });
    const call = client.WatchTelemetry({
      includeMotors: true,
      includeImu: true,
      includeHealth: true,
      includePowerBoards: true,
      includeCommandPolicyEvents: true,
    });
    call.on("data", (event) => {
      if (!current()) return;
      if (event.motor)
        snapshot.motors[event.motor.motorName] = {
          ...event.motor,
          receivedAt: Date.now(),
        };
      const received = (payload) => ({
        ...payload,
        receivedAt: Date.now(),
        monotonicNs: event.monotonicNs,
      });
      if (event.group)
        snapshot.groups[event.group.group] = received(event.group);
      if (event.imu) {
        snapshot.imu = received(event.imu);
        snapshot.imus[event.imu.imuName] = snapshot.imu;
      }
      if (event.health)
        snapshot.health[event.health.component] = received(event.health);
      if (event.powerBoard)
        snapshot.powerBoards[event.powerBoard.boardName] = received(
          event.powerBoard,
        );
      if (event.commandPolicy) {
        snapshot.commandPolicyEvents.unshift(received(event.commandPolicy));
        snapshot.commandPolicyEvents.length = Math.min(
          100,
          snapshot.commandPolicyEvents.length,
        );
      }
      mark("hardware");
    });
    return call;
  });
}
function startEstimator() {
  const api = schema(
    "pose.proto",
    "deps/bello-stack/components/bello-pose-api/proto/bello/pose/v1/pose.proto",
  );
  const client = new api.bello.pose.v1.PoseService(
    config.estimator,
    grpc.credentials.createInsecure(),
  );
  clients.add(client);
  reconnectingStream("estimator", (current) => {
    const call = client.WatchPose({
      referenceFrame: config.referenceFrame,
      trackedFrame: config.trackedFrame,
    });
    call.on("data", (pose) => {
      if (!current()) return;
      if (
        pose.referenceFrame !== config.referenceFrame ||
        pose.trackedFrame !== config.trackedFrame
      ) {
        mark("estimator", "Pose frame mismatch");
        return;
      }
      snapshot.pose = pose;
      mark("estimator");
    });
    return call;
  });
}
function startOperator() {
  const api = schema(
    "operator.proto",
    "rust/bello-policy/proto/operator.proto",
  );
  const client = new api.bello.operator.v1.OperatorService(
    config.runner,
    grpc.credentials.createInsecure(),
  );
  clients.add(client);
  reconnectingStream("runner", (current) => {
    const call = client.Exchange();
    operatorCall = call;
    call.on("data", (status) => {
      if (!current()) return;
      snapshot.operator = status;
      mark("runner");
      if (
        pendingClear?.sequence &&
        status.clearFaultSequence === pendingClear.sequence
      ) {
        if (status.mode === "MODE_DISARMED") finishClear();
        else if (status.recoveryError) finishClear(status.recoveryError);
      }
    });
    const clear = () => {
      if (operatorCall === call) {
        operatorCall = null;
        cancelAbortHold();
        pendingAbort = false;
        pendingSelection = null;
        finishClear(
          "Runner disconnected while clearing fault. Check its status before retrying.",
        );
      }
    };
    call.on("end", clear);
    call.on("error", clear);
    return call;
  });
}

function startTracking() {
  const api = schema(
    "pose_stream.proto",
    "deps/pico-smplx-bridge/pose_stream.proto",
  );
  const client = new api.pico.smplx.v1.PoseStream(
    config.tracking,
    grpc.credentials.createInsecure(),
    { "grpc.max_receive_message_length": 1024 * 1024 },
  );
  clients.add(client);
  reconnectingStream("tracking", (current) => {
    const call = client.StreamPoses({ maxRateHz: 30, includeRaw: false });
    call.on("data", (value) => {
      if (!current()) return;
      try {
        snapshot.tracking = trackingFrame(value);
        mark("tracking");
      } catch (error) {
        snapshot.tracking = null;
        mark("tracking", error.message);
      }
    });
    return call;
  });
}

function stopConnections() {
  cancelAbortHold();
  pendingAbort = false;
  pendingSelection = null;
  finishClear(
    "Connections changed while clearing fault. Check runner status before retrying.",
  );
  connectionGeneration += 1;
  operatorCall = null;
  snapshot.lastSentSequence = null;
  for (const timer of retryTimers) clearTimeout(timer);
  retryTimers.clear();
  for (const stream of streams) stream.cancel();
  streams.clear();
  for (const client of clients) client.close();
  clients.clear();
}
function startConnections() {
  startHardware();
  startEstimator();
  startOperator();
  startTracking();
}

function isConsoleSender(event) {
  return (
    window &&
    !window.isDestroyed() &&
    event.sender === window.webContents &&
    event.senderFrame === window.webContents.mainFrame
  );
}

function runnerReady(call = operatorCall) {
  return Boolean(
    call &&
    call === operatorCall &&
    !call.destroyed &&
    snapshot.services.runner.connected &&
    Date.now() - snapshot.services.runner.last < config.freshnessMs,
  );
}

app.whenReady().then(() => {
  ipcMain.handle("select-choice", (event, kind, name) => {
    if (
      !isConsoleSender(event) ||
      !["mode", "motion", "motionMenuOpen", "trackingEnabled"].includes(kind) ||
      typeof name !==
        (["motionMenuOpen", "trackingEnabled"].includes(kind)
          ? "boolean"
          : "string")
    )
      return { error: "Invalid selection." };
    if (!runnerReady()) return { error: "Runner unavailable." };
    const status = snapshot.operator;
    if (kind === "trackingEnabled") {
      if (
        status?.mode !== "MODE_LOCOMOTION" ||
        !status.trackingAvailable ||
        !snapshot.controller.connected ||
        status.neutralRequired
      )
        return {
          error:
            "Upper-body tracking requires Locomotion and a ready controller.",
        };
      pendingSelection = { trackingEnabled: name };
      return {};
    }
    if (status?.mode !== "MODE_STAND") return { error: "Select in Stand." };
    const choices =
      kind === "mode"
        ? status.modes
        : status.selected === "Motion"
          ? status.motions
          : [];
    if (
      kind === "motionMenuOpen"
        ? status.selected !== "Motion"
        : !choices?.includes(name)
    )
      return { error: "Choice unavailable." };
    pendingSelection = { [kind]: name };
    return {};
  });
  ipcMain.handle("toggle-fullscreen", (event) => {
    if (!isConsoleSender(event)) return;
    window.setFullScreen(!window.isFullScreen());
  });
  ipcMain.handle("abort-hold", (event, held) => {
    if (!isConsoleSender(event) || typeof held !== "boolean")
      return { error: "Invalid abort request." };
    if (!held) {
      cancelAbortHold();
      return {};
    }
    if (abortHoldTimer) return {};
    const call = operatorCall;
    if (!runnerReady(call)) return { error: "Runner unavailable." };
    abortHoldTimer = setTimeout(() => {
      // Keep the timer set until release so key repeats cannot retrigger it.
      if (runnerReady(call)) pendingAbort = true;
    }, 1000);
    return {};
  });
  ipcMain.handle("clear-fault", (event) => {
    if (!isConsoleSender(event))
      return { error: "Fault clearing is only available in the console." };
    if (pendingClear)
      return { error: "Fault clearing is already in progress." };
    if (!runnerReady() || !snapshot.operator?.canClearFault)
      return {
        error: "Wait for the runner to finish fault cleanup before clearing.",
      };
    return new Promise((resolve) => {
      pendingClear = {
        resolve,
        sequence: null,
        timer: setTimeout(
          () =>
            finishClear(
              "Runner did not confirm fault clearing. Check its status before retrying.",
            ),
          10000,
        ),
      };
    });
  });
  ipcMain.handle("save-settings", (event, value) => {
    if (!isConsoleSender(event))
      return {
        error: "Settings are only available in the console.",
      };
    let next;
    try {
      next = validateSettings(value);
    } catch (error) {
      return { error: error.message };
    }
    try {
      fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
      fs.writeFileSync(
        settingsPath + ".tmp",
        JSON.stringify({ ...config, ...next }, null, 2) + "\n",
      );
      fs.renameSync(settingsPath + ".tmp", settingsPath);
    } catch {
      return {
        error:
          "Could not save settings. Check that the configuration file is writable.",
      };
    }
    const changed = Object.keys(snapshot.endpoints).some(
      (key) => next[key] !== config[key],
    );
    Object.assign(config, next);
    snapshot.controllerStyle = next.controllerStyle;
    snapshot.theme = next.theme;
    if (!changed) return { settings: next };
    stopConnections();
    snapshot.endpoints = validateConnections(next);
    for (const service of Object.keys(snapshot.services))
      snapshot.services[service] = { connected: false, last: 0, error: "" };
    Object.assign(snapshot, emptyTelemetry());
    startConnections();
    return { settings: next };
  });
  const assets = resource(
    "bello_mini",
    "deps/bello-stack/descriptions/bello_mini",
  );
  protocol.handle("bello", (request) => {
    const url = new URL(request.url);
    const file = path.resolve(assets, "." + decodeURIComponent(url.pathname));
    if (url.host !== "assets" || !file.startsWith(assets + path.sep))
      return new Response("Not found", { status: 404 });
    return net.fetch(pathToFileURL(file).toString());
  });
  window = new BrowserWindow({
    icon: path.join(__dirname, "assets/bello-face.png"),
    width: 1440,
    height: 900,
    minWidth: 375,
    minHeight: 360,
    resizable: true,
    maximizable: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  window.webContents.session.setPermissionRequestHandler(
    (_contents, _permission, callback) => callback(false),
  );
  window.loadFile(path.join(__dirname, "dist/index.html"));
  startController();
  startConnections();
  window.on("blur", cancelAbortHold);
  every(() => {
    const sample = controllerSample();
    if (pendingClear && !pendingClear.sequence) {
      sample.clearFault = true;
      pendingClear.sequence = sample.sequence;
    }
    if (pendingAbort && operatorCall && !operatorCall.destroyed) {
      sample.abort = true;
      pendingAbort = false;
    }
    if (pendingSelection) {
      if (
        runnerReady() &&
        snapshot.operator?.mode ===
          ("trackingEnabled" in pendingSelection
            ? "MODE_LOCOMOTION"
            : "MODE_STAND") &&
        !sample.abort
      )
        sample.selection = pendingSelection;
      pendingSelection = null;
    }
    snapshot.controllerSample = sample;
    if (operatorCall && !operatorCall.destroyed) {
      snapshot.lastSentSequence = sample.sequence;
      if (!operatorCall.write(sample)) operatorCall.cancel();
    }
  }, 20);
  every(() => {
    if (!window || window.isDestroyed()) return;
    const state = displaySnapshot(snapshot, config);
    state.fullscreen = window.isFullScreen();
    window.webContents.send("snapshot", state);
  }, 100);
});
app.on("window-all-closed", () => app.quit());
app.on("before-quit", () => {
  quitting = true;
  for (const timer of timers) {
    clearTimeout(timer);
    clearInterval(timer);
  }
  stopConnections();
  if (controller && !controller.closed) controller.close();
});
