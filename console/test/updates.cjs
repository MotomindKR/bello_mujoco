const assert = require("node:assert/strict");
const test = require("node:test");
const {
  linuxPackageKind,
  releaseAsset,
  expectedHash,
} = require("../updates.cjs");

test("selects only the matching newer release package and checksum", () => {
  const base =
    "https://github.com/MotomindKR/bello_mujoco/releases/download/console-v0.2.9/";
  const assets = ["linux-x64.deb", "linux-x64.rpm", "linux-arm64.deb"].map(
    (suffix) => {
      const name = `Bello-Mini-Console-0.2.9-${suffix}`;
      return { name, browser_download_url: base + name, size: 123 };
    },
  );
  assets.push({
    name: "SHA256SUMS",
    browser_download_url: base + "SHA256SUMS",
  });
  const release = { tag_name: "console-v0.2.9", assets };
  const candidate = releaseAsset(release, "0.2.8", "linux", "x64", "deb");
  assert.equal(candidate.name, "Bello-Mini-Console-0.2.9-linux-x64.deb");
  assert.equal(candidate.checksumUrl, base + "SHA256SUMS");
  assert.equal(releaseAsset(release, "0.2.9", "linux", "x64", "deb"), null);
  assert.equal(
    releaseAsset(
      {
        ...release,
        assets: [
          { ...assets[0], browser_download_url: "https://evil.test/package" },
          assets[3],
        ],
      },
      "0.2.8",
      "linux",
      "x64",
      "deb",
    ).unavailable,
    true,
  );
});

test("chooses native package formats and exact checksum entries", () => {
  assert.equal(linuxPackageKind("ID=ubuntu\nID_LIKE=debian\n"), "deb");
  assert.equal(linuxPackageKind('ID="fedora"\n'), "rpm");
  assert.equal(linuxPackageKind("ID=arch\n"), null);
  const hash = "a".repeat(64);
  assert.equal(expectedHash(`${hash}  package.deb\n`, "package.deb"), hash);
  assert.equal(expectedHash(`${hash}  other.deb\n`, "package.deb"), null);
});
