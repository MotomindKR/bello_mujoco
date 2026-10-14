const assert = require("node:assert/strict");
const { test } = require("node:test");
const { EventEmitter } = require("node:events");
const {
  sampleController,
  createTabNavigator,
  createOperatorMapper,
} = require("../controller.cjs");

function device() {
  return Object.assign(new EventEmitter(), {
    closed: false,
    device: { name: "Test controller" },
    axes: { leftStickY: -1, leftStickX: 1, rightStickX: -1 },
    buttons: {
      back: true,
      start: true,
      a: true,
      rightShoulder: true,
      dpadUp: true,
      dpadDown: true,
    },
  });
}

for (const property of ["axes", "buttons"]) {
  test(`disconnect during ${property} polling returns neutral input and permits reconnect`, () => {
    let active = device();
    const original = active;
    const values = original[property];
    original.on("close", () => {
      active = null;
    });
    Object.defineProperty(original, property, {
      get() {
        original.emit("close");
        original.closed = true;
        return values;
      },
    });
    assert.deepEqual(
      sampleController(active, 0.12, "positive"),
      sampleController(null, 0.12, "positive"),
    );
    assert.equal(active, null);
    active = device();
    const reconnected = sampleController(active, 0.12, "positive");
    assert.equal(reconnected.connected, true);
    assert.equal(reconnected.name, "Test controller");
    assert.equal(reconnected.forward, 1);
    assert.equal(reconnected.lateral, 1);
    assert.equal(reconnected.yaw, 1);
  });
}

test("closed controllers are never polled", () => {
  const closed = {
    closed: true,
    get axes() {
      throw new Error("must not poll");
    },
  };
  assert.deepEqual(
    sampleController(closed, 0.12, "positive"),
    sampleController(null, 0.12, "positive"),
  );
});

test("shoulder buttons are reported independently for tab navigation", () => {
  const controller = device();
  for (const [leftShoulder, rightShoulder] of [
    [true, false],
    [false, true],
    [true, true],
  ]) {
    controller.buttons = { leftShoulder, rightShoulder };
    const sample = sampleController(controller, 0.12, "positive");
    assert.equal(sample.l1, leftShoulder);
    assert.equal(sample.r1, rightShoulder);
  }
});

test("trigger axes are reported independently above half travel", () => {
  const controller = device();
  for (const [leftTrigger, rightTrigger, l2, r2] of [
    [0, 0, false, false],
    [0.5, 0.51, false, true],
    [0.75, 0.25, true, false],
    [1, 1, true, true],
  ]) {
    controller.axes = { leftTrigger, rightTrigger };
    const sample = sampleController(controller, 0.12, "positive");
    assert.equal(sample.l2, l2);
    assert.equal(sample.r2, r2);
  }
  assert.equal(sampleController(null, 0.12, "positive").l2, false);
  assert.equal(sampleController(null, 0.12, "positive").r2, false);
});

test("inverted Xbox trigger readings require a physical press to abort", () => {
  const controller = device();
  controller.buttons = {};
  const released = 0.5000076295109483;
  const pressed = -0.49999237048905165;
  const map = createOperatorMapper();
  for (const [leftTrigger, rightTrigger, l2, r2, abort] of [
    [released, released, false, false, false],
    [pressed, released, true, false, false],
    [released, pressed, false, true, false],
    [pressed, pressed, true, true, true],
  ]) {
    controller.axes = { leftTrigger, rightTrigger };
    const sample = sampleController(controller, 0.12, "negative");
    assert.equal(sample.l2, l2);
    assert.equal(sample.r2, r2);
    assert.equal(map(sample, { mode: "MODE_STAND" }).abort, abort);
  }
});

test("horizontal D-pad buttons are sampled independently", () => {
  const controller = device();
  for (const [dpadLeft, dpadRight] of [
    [true, false],
    [false, true],
    [true, true],
  ]) {
    controller.buttons = { dpadLeft, dpadRight };
    const sample = sampleController(controller, 0.12, "positive");
    assert.equal(sample.dpadLeft, dpadLeft);
    assert.equal(sample.dpadRight, dpadRight);
    assert.equal(sample.dpadUp, false);
    assert.equal(sample.dpadDown, false);
  }
});
test("Y is sampled for the console tracking toggle", () => {
  const controller = device();
  controller.buttons.y = true;
  assert.equal(sampleController(controller, 0.12, "positive").y, true);
  assert.equal(sampleController(null, 0.12, "positive").y, false);
});
test("operator mapping streams movement and semantic holds", () => {
  const map = createOperatorMapper();
  const raw = sampleController(null, 0.12, "positive");
  const status = { mode: "MODE_STAND", selected: "Motion" };
  assert.deepEqual(
    {
      forward: map({ ...raw, forward: 0.5 }, status).forward,
      requestMode: map({ ...raw, connected: true, a: true }, status)
        .requestMode,
      requestHold: map(
        { ...raw, connected: true, back: true, start: true },
        status,
      ).requestHold,
    },
    { forward: 0.5, requestMode: "Motion", requestHold: true },
  );
  assert.equal(
    map({ ...raw, connected: true, a: true }, { mode: "MODE_FAULT" })
      .confirmFaultRecovery,
    true,
  );
  assert.equal(map(raw, status, true).abort, true);
  assert.equal(map(raw, status, true).abortOtherControlsNeutral, true);
});
test("selection edges and held controls remain distinct", () => {
  const map = createOperatorMapper();
  const raw = { ...sampleController(null, 0.12, "positive"), connected: true };
  const status = { mode: "MODE_STAND", selected: "Locomotion" };
  const first = map({ ...raw, dpadRight: true, a: true }, status);
  assert.equal(first.modeStep, 1);
  assert.equal(first.navigating, true);
  assert.equal(first.requestMode, "Locomotion");
  assert.equal(map({ ...raw, dpadRight: true, a: true }, status).modeStep, 0);
  assert.equal(map(raw, status).controlsNeutral, true);
  assert.equal(map({ ...raw, l2: true }, status).controlsNeutral, false);
  assert.equal(map({ ...raw, l2: true, r2: true }, status).abort, true);
});
test("bumper presses cycle tabs once per click", () => {
  const navigate = createTabNavigator();
  const sample = (l1, r1, connected = true) => ({ l1, r1, connected });
  assert.equal(navigate(sample(true, false)), -1);
  assert.equal(navigate(sample(true, false)), 0);
  assert.equal(navigate(sample(false, false)), 0);
  assert.equal(navigate(sample(false, true)), 1);
  assert.equal(navigate(sample(false, false)), 0);
  assert.equal(navigate(sample(true, true)), 0);
  assert.equal(navigate(sample(false, true)), 0);
  assert.equal(navigate(sample(false, false)), 0);
  assert.equal(navigate(sample(true, false)), -1);
  assert.equal(navigate(sample(true, true)), 1);
  assert.equal(navigate(sample(true, false)), 0);
  assert.equal(navigate(sample(false, false)), 0);
  assert.equal(navigate(sample(true, false)), -1);
  assert.equal(navigate(sample(false, false, false)), 0);
  assert.equal(navigate(sample(true, false)), -1);
});

test("Barbell maps confirmations to the current generation and Y to cancellation", () => {
  const map = createOperatorMapper();
  const sample = {
    ...sampleController(null, 0.12, "positive"),
    connected: true,
    a: true,
  };
  const status = {
    mode: "MODE_BARBELL",
    barbell: {
      awaiting: "bar_supported",
      phaseId: "12",
      confirmationReady: true,
    },
  };
  const input = map(sample, status);
  assert.equal(input.confirmBarbell, "bar_supported");
  assert.equal(input.barbellPhaseId, "12");
  assert.equal(input.requestMode, "");
  assert.equal(
    map(sample, {
      ...status,
      barbell: { ...status.barbell, confirmationReady: false },
    }).confirmBarbell,
    "",
  );
  const cancel = map({ ...sample, a: false, y: true }, status);
  assert.equal(cancel.requestBarbellStop, true);
  assert.equal(cancel.toggleTracking, false);
  assert.equal(
    map({ ...sample, l2: true, r2: true }, status).confirmBarbell,
    "",
  );
});
