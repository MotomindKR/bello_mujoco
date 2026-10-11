const fs = require("node:fs");

const releasesUrl = "https://github.com/MotomindKR/bello_mujoco/releases";
const latestReleaseUrl =
  "https://api.github.com/repos/MotomindKR/bello_mujoco/releases/latest";

function versionParts(version) {
  if (!/^\d+\.\d+\.\d+$/.test(version)) return null;
  return version.split(".").map(Number);
}

function newerVersion(candidate, current) {
  const next = versionParts(candidate);
  const installed = versionParts(current);
  if (!next || !installed) return false;
  for (let i = 0; i < next.length; i++) {
    if (next[i] !== installed[i]) return next[i] > installed[i];
  }
  return false;
}

function linuxPackageKind(
  osRelease = fs.readFileSync("/etc/os-release", "utf8"),
) {
  const fields = Object.fromEntries(
    osRelease
      .split("\n")
      .map((line) => /^([A-Z_]+)=(.*)$/.exec(line))
      .filter(Boolean)
      .map(([, key, value]) => [key, value.replace(/^"|"$/g, "")]),
  );
  const ids = `${fields.ID || ""} ${fields.ID_LIKE || ""}`.split(/\s+/);
  if (ids.some((id) => ["debian", "ubuntu"].includes(id))) return "deb";
  if (ids.some((id) => ["fedora", "rhel", "centos"].includes(id))) return "rpm";
  return null;
}

function releaseAsset(release, current, platform, arch, packageKind) {
  const match = /^console-v(\d+\.\d+\.\d+)$/.exec(release?.tag_name || "");
  if (!match || !newerVersion(match[1], current)) return null;
  if (platform !== "linux" || !["deb", "rpm"].includes(packageKind))
    return { version: match[1], unsupported: true };
  const target = arch === "x64" ? "x64" : arch === "arm64" ? "arm64" : null;
  if (!target) return { version: match[1], unsupported: true };
  const name = `Bello-Mini-Console-${match[1]}-linux-${target}.${packageKind}`;
  const prefix = `${releasesUrl}/download/${release.tag_name}/`;
  const asset = release.assets?.find(
    (item) => item.name === name && item.browser_download_url === prefix + name,
  );
  const checksum = release.assets?.find(
    (item) =>
      item.name === "SHA256SUMS" &&
      item.browser_download_url === prefix + "SHA256SUMS",
  );
  if (!asset || !checksum) return { version: match[1], unavailable: true };
  return {
    version: match[1],
    name,
    packageKind,
    size: asset.size,
    url: asset.browser_download_url,
    checksumUrl: checksum.browser_download_url,
  };
}

function expectedHash(checksums, filename) {
  for (const line of checksums.split("\n")) {
    const match = /^([a-fA-F0-9]{64})  (\S+)$/.exec(line.trimEnd());
    if (match && match[2] === filename) return match[1].toLowerCase();
  }
  return null;
}

module.exports = {
  releasesUrl,
  latestReleaseUrl,
  linuxPackageKind,
  releaseAsset,
  expectedHash,
};
