{
  description = "Minimal Nix shell for a uv-managed MuJoCo/MJX robot-learning stack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem
      [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ]
      (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        mujocoGl = if pkgs.stdenv.isDarwin then "glfw" else "egl";
        evalPolicyViewer = pkgs.writeShellApplication {
          name = "bello-eval-policy-viewer";
          text = ''
            if [ "$(uname -s)" = "Darwin" ]; then
              if ! command -v mjpython >/dev/null 2>&1; then
                echo "mjpython not found; enter nix develop so uv sync can activate .venv" >&2
                exit 127
              fi
              exec mjpython -m bello_mujoco.tools.eval_policy "$@"
            fi

            exec bello-eval-policy "$@"
          '';
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            evalPolicyViewer
            pkgs.git
            pkgs.uv
          ];

          UV_PROJECT_ENVIRONMENT = ".venv";

          shellHook = ''
            export UV_CACHE_DIR="$PWD/.uv-cache"
            export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
            export WARP_CACHE_PATH="$PWD/.warp-cache"
            export MUJOCO_GL=${mujocoGl}
            export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"

            uv python install 3.12
            uv sync

            if [ -f .venv/bin/activate ]; then
              source .venv/bin/activate
            fi
          '';
        };
      }
    );
}
