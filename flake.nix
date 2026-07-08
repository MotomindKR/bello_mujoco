{
  description = "Minimal Nix shell for a uv-managed MuJoCo/MJX robot-learning stack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.git
            pkgs.uv
          ];

          UV_PROJECT_ENVIRONMENT = ".venv";

          shellHook = ''
            export UV_CACHE_DIR="$PWD/.uv-cache"
            export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
            export WARP_CACHE_PATH="$PWD/.warp-cache"
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
