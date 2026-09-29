const path = require("node:path");
const root = path.resolve(__dirname, "..");
module.exports = {
  packagerConfig: {
    name: "Bello Mini Console",
    appBundleId: "kr.motomind.bello-mini-console",
    icon: path.join(
      __dirname,
      process.platform === "darwin"
        ? "out/bello-face.icns"
        : "assets/bello-face.png",
    ),
    ignore: [/^\/test(?:\/|$)/, /^\/src(?:\/|$)/, /^\/scripts(?:\/|$)/],
    executableName: "bello-mini-console",
    asar: { unpackDir: "node_modules/@kmamal/sdl" },
    extraResource: [
      path.join(root, "rust/bello-policy/proto/operator.proto"),
      path.join(root, "deps/pico-smplx-bridge/pose_stream.proto"),
      path.join(
        root,
        "deps/bello-stack/components/bello-hardware-api/proto/bello/hardware/v1/hardware.proto",
      ),
      path.join(
        root,
        "deps/bello-stack/components/bello-pose-api/proto/bello/pose/v1/pose.proto",
      ),
      path.join(root, "deps/bello-stack/descriptions/bello_mini"),
    ],
  },
  rebuildConfig: { onlyModules: ["@kmamal/sdl"] },
  hooks: { prePackage: async () => require("./scripts/build-icons.cjs")() },
  makers: [
    { name: "@electron-forge/maker-dmg", platforms: ["darwin"] },
    { name: "@electron-forge/maker-zip", platforms: ["darwin"] },
    {
      name: "@reforged/maker-appimage",
      platforms: ["linux"],
      config: {
        options: {
          bin: "bello-mini-console",
          icon: path.join(__dirname, "assets/bello-face.png"),
          categories: ["Science", "Engineering"],
        },
      },
    },
  ],
};
