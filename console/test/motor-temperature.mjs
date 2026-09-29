import assert from "node:assert/strict";
import test from "node:test";
import { motorTemperature } from "../src/motor-temperature.mjs";
const state = (motors, fresh = true) => ({
  services: { hardware: { fresh } },
  motors,
});
const motor = (temperature, fresh = true) => ({ temperature, fresh });
test("temperature threshold is strictly above 60 and chooses the hottest motor", () => {
  assert.equal(motorTemperature(state({ a: motor(60) })).hot, false);
  const hottest = motor(70);
  assert.deepEqual(motorTemperature(state({ a: motor(60.01), b: hottest })), {
    hottest,
    hot: true,
    incomplete: false,
  });
});
test("missing, non-finite and stale telemetry cannot be called nominal", () => {
  for (const value of [
    state({}),
    state({ a: motor(20, false) }),
    state({ a: motor(NaN) }),
    state({ a: motor(20) }, false),
  ]) {
    assert.equal(motorTemperature(value).incomplete, true);
  }
  const value = state({ a: motor(20) });
  value.inventory = { motors: [{ name: "a" }, { name: "b" }] };
  assert.equal(motorTemperature(value).incomplete, true);
  assert.equal(
    motorTemperature(state({ a: motor(80), b: motor(20, false) })).hot,
    true,
  );
});
