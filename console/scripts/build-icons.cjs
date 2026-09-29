const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

module.exports = function buildIcons() {
  if (process.platform !== "darwin") return;
  const root = path.resolve(__dirname, "..");
  const source = path.join(root, "assets/bello-face.png");
  const iconset = path.join(root, "out/bello-face.iconset");
  fs.mkdirSync(iconset, { recursive: true });
  for (const size of [16, 32, 128, 256, 512]) {
    for (const scale of [1, 2]) {
      execFileSync(
        "sips",
        [
          "-z",
          String(size * scale),
          String(size * scale),
          source,
          "--out",
          path.join(
            iconset,
            `icon_${size}x${size}${scale === 2 ? "@2x" : ""}.png`,
          ),
        ],
        { stdio: "pipe" },
      );
    }
  }
  execFileSync("iconutil", [
    "-c",
    "icns",
    iconset,
    "-o",
    path.join(root, "out/bello-face.icns"),
  ]);
};
