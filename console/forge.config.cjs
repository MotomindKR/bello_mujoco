const path = require("node:path");
const root = path.resolve(__dirname, "..");
module.exports = {
  packagerConfig: {
    name: "Bello Mini Console",
    appBundleId: "kr.motomind.bello-mini-console",
    extendInfo: {
      NSLocalNetworkUsageDescription:
        "Connect to Bello robots and tracking services on your local network.",
    },
    osxSign: {
      identity: "-",
      identityValidation: false,
      preAutoEntitlements: false,
      preEmbedProvisioningProfile: false,
      gatekeeperAssess: false,
      optionsForFile: () => ({
        entitlements: ["com.apple.security.cs.allow-jit"],
        hardenedRuntime: false,
      }),
    },
    icon: path.join(
      __dirname,
      process.platform === "darwin"
        ? "out/bello-face.icns"
        : "assets/bello-face.png",
    ),
    ignore: [/^\/test(?:\/|$)/, /^\/src(?:\/|$)/, /^\/scripts(?:\/|$)/],
    executableName: "bello-mini-console",
    asar: { unpackDir: "node_modules/@kmamal/sdl" },
    afterAsar: [require("./scripts/mac-identity.cjs").afterAsar],
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
      name: "@electron-forge/maker-deb",
      platforms: ["linux"],
      config: {
        options: {
          name: "bello-mini-console",
          description: "Bello Mini robot operator console",
          maintainer: "Motomind",
          homepage: "https://github.com/MotomindKR/bello_mujoco",
          bin: "bello-mini-console",
          icon: path.join(__dirname, "assets/bello-face.png"),
          desktopTemplate: path.join(
            __dirname,
            "assets/bello-mini-console.desktop",
          ),
        },
      },
    },
    {
      name: "@electron-forge/maker-rpm",
      platforms: ["linux"],
      config: {
        options: {
          name: "bello-mini-console",
          description: "Bello Mini robot operator console",
          homepage: "https://github.com/MotomindKR/bello_mujoco",
          license: "Proprietary",
          bin: "bello-mini-console",
          icon: path.join(__dirname, "assets/bello-face.png"),
          desktopTemplate: path.join(
            __dirname,
            "assets/bello-mini-console.desktop",
          ),
        },
      },
    },
  ],
};
