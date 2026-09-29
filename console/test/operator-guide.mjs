import assert from "node:assert/strict";
import test from "node:test";
import { operatorGuide } from "../src/operator-guide.mjs";
const state = (operator = {}, services = {}) => ({
  services: { runner: { fresh: true, ...services } },
  controller: { connected: true },
  operator: { controllerConnected: true, neutralRequired: false, ...operator },
});
test("stale state cannot offer an arming action", () => {
  const guide = operatorGuide(
    state({ mode: "MODE_DISARMED", phase: "disarmed" }, { fresh: false }),
  );
  assert.equal(guide.keys, undefined);
  assert.equal(guide.title, "Runner offline");
});
test("recovery and neutral input take priority over mode actions", () => {
  assert.equal(
    operatorGuide(
      state({ mode: "MODE_FAULT", canClearFault: true, neutralRequired: true }),
    ).title,
    "Control stopped",
  );
  assert.equal(
    operatorGuide(state({ mode: "MODE_STAND", neutralRequired: true })).keys,
    undefined,
  );
  assert.match(
    operatorGuide(state({ controllerConnected: false })).title,
    /controller link/,
  );
});
test("instructions follow runner phases and transition restrictions", () => {
  assert.deepEqual(
    operatorGuide(state({ mode: "MODE_DISARMED", phase: "disarmed" })).keys,
    ["Back", "Start"],
  );
  assert.equal(
    operatorGuide(state({ mode: "MODE_HOLD", phase: "ramping" })).keys,
    undefined,
  );
  assert.match(
    operatorGuide(state({ mode: "MODE_HOLD", phase: "holding" })).duration,
    /Stand/,
  );
  assert.match(
    operatorGuide(
      state({
        mode: "MODE_STAND",
        selected: "Motion",
        selectedMotion: "Lunge",
      }),
    ).duration,
    /Lunge/,
  );
  assert.equal(
    operatorGuide(
      state({ mode: "MODE_STAND", selected: "Motion", motionMenuOpen: true }),
    ).title,
    "Choose motion",
  );
  assert.match(
    operatorGuide(state({ mode: "MODE_LOCOMOTION" })).duration,
    /Stand/,
  );
  assert.equal(
    operatorGuide(state({ mode: "MODE_MOTION", activeMotion: "Lunge" })).keys,
    undefined,
  );
});
test("buffered Tracking offers entry only when the runner is ready", () => {
  assert.equal(
    operatorGuide(
      state({
        mode: "MODE_STAND",
        selected: "Tracking",
        sonicReady: false,
        sonicStatus: "Buffering reference",
      }),
    ).keys,
    undefined,
  );
  assert.deepEqual(
    operatorGuide(
      state({ mode: "MODE_STAND", selected: "Tracking", sonicReady: true }),
    ).keys,
    ["A"],
  );
  assert.match(
    operatorGuide(state({ mode: "MODE_TRACKING", phase: "motion" })).duration,
    /Stand/,
  );
});

test("assisted startup guidance follows balance and ankle phases", () => {
  const hold = (phase, startupReady = false) =>
    operatorGuide(state({ mode: "MODE_HOLD", phase, startupReady }));
  assert.equal(hold("bending").keys, undefined);
  assert.equal(hold("awaiting lift").keys, undefined);
  assert.match(hold("awaiting lift").title, /Lift/);
  assert.match(hold("awaiting lift", true).duration, /Balanced/);
  assert.equal(hold("straightening").keys, undefined);
  assert.match(hold("straightening").title, /Left stick/);
  assert.match(hold("upright", true).duration, /Stand/);
  assert.equal(hold("engaging ankles").keys, undefined);
});
