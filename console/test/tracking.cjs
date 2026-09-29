const assert = require("node:assert/strict");
const test = require("node:test");
const { trackingFrame, bodyNames, handNames } = require("../tracking.cjs");
const { displaySnapshot, emptyTelemetry } = require("../snapshot.cjs");
const part = (names, hand = false) => ({
  frame: "pico_native_y_up",
  valid: true,
  active: true,
  ageSeconds: 0.02,
  jointNames: names,
  positions: names.flatMap((_, i) => [i, i + 1, i + 2]),
  orientations: names.flatMap(() => [1, 0, 0, 0]),
  ...(hand
    ? {
        positionValid: names.map(() => true),
        orientationValid: names.map(() => true),
      }
    : {}),
});
const frame = () => ({
  upstreamConnected: true,
  packetAgeSeconds: 0.01,
  xrobotBody: part(bodyNames),
  xrobotLeftHand: part(handNames, true),
});
test("native tracking preserves Y-up coordinates, converts WXYZ, and keeps separate source ages", () => {
  const result = trackingFrame(frame());
  assert.deepEqual(result.body.joints[1].position, [1, 2, 3]);
  assert.deepEqual(result.body.joints[0].orientation, [0, 0, 0, 1]);
  assert.equal(result.body.ageMs, 20);
  assert.equal(result.packetAgeMs, 10);
  assert.equal(result.rightHand, null);
});
test("malformed tracking cannot reach the 3D renderer", () => {
  for (const change of [
    (body) => (body.frame = "other"),
    (body) => body.positions.pop(),
    (body) => (body.positions[0] = NaN),
    (body) => (body.orientations[0] = 2),
    (body) => (body.jointNames = Array(24).fill("Pelvis")),
  ]) {
    const value = frame();
    change(value.xrobotBody);
    assert.throws(() => trackingFrame(value), /Invalid XRoboToolkit/);
  }
});
test("invalid hand joints and unavailable body frames are hidden", () => {
  const value = frame();
  value.xrobotBody.valid = false;
  value.xrobotLeftHand.positionValid[0] = false;
  value.xrobotLeftHand.orientationValid[0] = false;
  value.xrobotLeftHand.positions[0] = NaN;
  const result = trackingFrame(value);
  assert.equal(result.body, null);
  assert.equal(result.leftHand.joints[0].position, null);
  assert.equal(result.leftHand.joints[0].orientation, null);
});

test("the bridge's pre-packet sentinel is connected but never fresh", () => {
  const tracking = trackingFrame({
    sequence: "0",
    upstreamConnected: true,
    packetAgeSeconds: -1,
  });
  assert.equal(tracking.packetAgeMs, null);
  assert.equal(tracking.body, null);
  const state = displaySnapshot(
    {
      ...emptyTelemetry(),
      tracking,
      services: { tracking: { connected: true, last: 1000 } },
    },
    { freshnessMs: 250 },
    1010,
  );
  assert.equal(state.services.tracking.connected, true);
  assert.equal(state.services.tracking.fresh, false);
  for (const age of [-1, -2, NaN])
    assert.throws(
      () =>
        trackingFrame({
          sequence: "1",
          upstreamConnected: true,
          packetAgeSeconds: age,
        }),
      /Invalid XRoboToolkit packet age/,
    );
});
