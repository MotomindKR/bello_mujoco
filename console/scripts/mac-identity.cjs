const { createHash } = require("node:crypto");
const fs = require("node:fs/promises");
const path = require("node:path");
const { callbackify } = require("node:util");

function uuidOffset(binary) {
  if (binary.length < 32 || binary.readUInt32LE(0) !== 0xfeedfacf)
    throw new Error("Expected a thin 64-bit Mach-O executable");
  const end = 32 + binary.readUInt32LE(20);
  if (end > binary.length) throw new Error("Truncated Mach-O load commands");
  let offset = 32;
  let found;
  for (let i = 0; i < binary.readUInt32LE(16); i++) {
    if (offset + 8 > end) throw new Error("Truncated Mach-O load command");
    const command = binary.readUInt32LE(offset);
    const size = binary.readUInt32LE(offset + 4);
    if (size < 8 || offset + size > end)
      throw new Error("Invalid Mach-O load command size");
    if (command === 0x1b) {
      if (size !== 24 || found !== undefined)
        throw new Error("Invalid or duplicate Mach-O UUID");
      found = offset + 8;
    }
    offset += size;
  }
  if (offset !== end || found === undefined)
    throw new Error("Missing Mach-O UUID or inconsistent load commands");
  return found;
}

function setAppUuid(binary, archive, bundleId) {
  const offset = uuidOffset(binary);
  // TN3179: sharing stock Electron's executable UUID confuses network privacy.
  const uuid = createHash("sha256")
    .update(bundleId)
    .update(binary.subarray(offset, offset + 16))
    .update(archive)
    .digest()
    .subarray(0, 16);
  uuid[6] = (uuid[6] & 0x0f) | 0x80;
  uuid[8] = (uuid[8] & 0x3f) | 0x80;
  uuid.copy(binary, offset);
}

async function afterAsar(buildPath, _version, platform) {
  if (platform !== "darwin") return;
  const executable = path.resolve(buildPath, "../../MacOS/Electron");
  const binary = await fs.readFile(executable);
  const archive = await fs.readFile(`${buildPath}.asar`);
  setAppUuid(binary, archive, "kr.motomind.bello-mini-console");
  // Packager signs this binary after the hook, binding its new UUID.
  await fs.writeFile(executable, binary);
}

module.exports = { afterAsar: callbackify(afterAsar), setAppUuid, uuidOffset };
