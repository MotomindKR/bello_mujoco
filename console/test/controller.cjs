const assert = require("node:assert/strict");
const { test } = require("node:test");
const { EventEmitter } = require("node:events");
const { sampleController } = require("../controller.cjs");

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
      sampleController(active, 0.12),
      sampleController(null, 0.12),
    );
    assert.equal(active, null);
    active = device();
    const reconnected = sampleController(active, 0.12);
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
    sampleController(closed, 0.12),
    sampleController(null, 0.12),
  );
});

test("shoulder buttons are reported independently for the runner abort chord", () => {
  const controller = device();
  for (const [leftShoulder, rightShoulder] of [
    [true, false],
    [false, true],
    [true, true],
  ]) {
    controller.buttons = { leftShoulder, rightShoulder };
    const sample = sampleController(controller, 0.12);
    assert.equal(sample.l1, leftShoulder);
    assert.equal(sample.r1, rightShoulder);
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
    const sample = sampleController(controller, 0.12);
    assert.equal(sample.dpadLeft, dpadLeft);
    assert.equal(sample.dpadRight, dpadRight);
    assert.equal(sample.dpadUp, false);
    assert.equal(sample.dpadDown, false);
  }
});
test("Y is sampled for the runner tracking toggle", () => {
  const controller = device();
  controller.buttons.y = true;
  assert.equal(sampleController(controller, 0.12).y, true);
  assert.equal(sampleController(null, 0.12).y, false);
});
