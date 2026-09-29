const assert = require("node:assert/strict");
const { test } = require("node:test");
const { setAppUuid, uuidOffset } = require("../scripts/mac-identity.cjs");

function executable(cpu) {
  const binary = Buffer.alloc(80, 0);
  binary.writeUInt32LE(0xfeedfacf, 0);
  binary.writeUInt32LE(cpu, 4);
  binary.writeUInt32LE(1, 16);
  binary.writeUInt32LE(24, 20);
  binary.writeUInt32LE(0x1b, 32);
  binary.writeUInt32LE(24, 36);
  binary.fill(0xab, 40, 56);
  return binary;
}

for (const cpu of [0x01000007, 0x0100000c]) {
  test(`app UUID isolates Electron builds without modifying code (${cpu})`, () => {
    const source = executable(cpu);
    const build = (app, archive) => {
      const result = Buffer.from(source);
      setAppUuid(result, Buffer.from(archive), app);
      return result;
    };
    const result = build("bello", "release-a");
    assert.notDeepEqual(result.subarray(40, 56), source.subarray(40, 56));
    assert.deepEqual(result.subarray(0, 40), source.subarray(0, 40));
    assert.deepEqual(result.subarray(56), source.subarray(56));
    assert.deepEqual(result, build("bello", "release-a"));
    assert.notDeepEqual(result, build("bello", "release-b"));
    assert.notDeepEqual(result, build("other-app", "release-a"));
  });
}

test("reject unsupported or malformed binaries before writing", () => {
  assert.throws(() => uuidOffset(Buffer.alloc(32)), /thin 64-bit/);
  const source = executable(0x0100000c);
  assert.throws(() => uuidOffset(source.subarray(0, 40)), /Truncated/);
  source.writeUInt32LE(0, 36);
  assert.throws(() => uuidOffset(source), /command size/);
  source.writeUInt32LE(24, 36);
  source.writeUInt32LE(0x19, 32);
  assert.throws(() => uuidOffset(source), /Missing/);
});
