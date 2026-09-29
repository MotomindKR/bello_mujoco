const assert = require("node:assert/strict");
const test = require("node:test");
const { emptyTelemetry, displaySnapshot } = require("../snapshot.cjs");

function fixture() {
  return {
    ...emptyTelemetry(),
    services: Object.fromEntries(
      ["hardware", "estimator", "runner", "tracking"].map((name) => [
        name,
        { connected: true, last: 1000 },
      ]),
    ),
    inventory: { powerBoards: [{ name: "power", staleTimeoutMs: "3000" }] },
    pose: { sampleAgeMs: "200" },
    motors: { motor: { receivedAt: 1000, feedbackAgeMs: "100" } },
    powerBoards: {
      power: {
        boardName: "power",
        receivedAt: 1000,
        samples: [
          { sampleAgeMs: "500", stale: false, battery: { socPercent: 80 } },
        ],
      },
    },
    tracking: {
      upstreamConnected: true,
      packetAgeMs: 10,
      body: { ageMs: 200 },
      leftHand: { ageMs: 10 },
    },
  };
}

test("display snapshots age packets without mutating source telemetry", () => {
  const source = fixture();
  const original = structuredClone(source);
  const first = displaySnapshot(source, { freshnessMs: 250 }, 1100);
  const second = displaySnapshot(source, { freshnessMs: 250 }, 1200);
  assert.equal(first.motors.motor.feedbackAgeMs, 200);
  assert.equal(second.motors.motor.feedbackAgeMs, 300);
  assert.equal(first.motors.motor.fresh, true);
  assert.equal(second.motors.motor.fresh, false);
  assert.equal(first.services.runner.fresh, true);
  assert.equal(first.services.estimator.fresh, false);
  assert.equal(first.tracking.body.fresh, false);
  assert.equal(first.tracking.leftHand.fresh, true);
  assert.equal(first.powerBoards.power.fresh, true);
  assert.deepEqual(source, original);
});

test("power freshness uses board timeout and preserves server stale flags", () => {
  const source = fixture();
  const config = { freshnessMs: 250 };
  let displayed = displaySnapshot(source, config, 1500);
  assert.equal(displayed.services.hardware.fresh, false);
  assert.equal(displayed.powerBoards.power.samples[0].stale, false);
  source.powerBoards.power.samples[0].stale = true;
  assert.equal(
    displaySnapshot(source, config, 1500).powerBoards.power.samples[0].stale,
    true,
  );
  source.powerBoards.power.samples[0].stale = false;
  assert.equal(
    displaySnapshot(source, config, 3500).powerBoards.power.samples[0].stale,
    true,
  );
  source.services.hardware.connected = false;
  assert.equal(
    displaySnapshot(source, config, 1100).powerBoards.power.samples[0].stale,
    true,
  );
});

test("reconnecting clears telemetry without sharing mutable containers", () => {
  const first = emptyTelemetry();
  first.motors.motor = {};
  first.commandPolicyEvents.push({});
  assert.deepEqual(emptyTelemetry().motors, {});
  assert.deepEqual(emptyTelemetry().commandPolicyEvents, []);
});
