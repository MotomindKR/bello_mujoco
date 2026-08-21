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
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.git
            pkgs.socat
            pkgs.uv
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.iproute2
            pkgs.kmod
          ];

          UV_PROJECT_ENVIRONMENT = ".venv";

          shellHook = ''
            if [ ! -f deps/bello-stack/components/bello-hardware-python/pyproject.toml ]; then
              echo "The pinned bello-stack submodule is not initialized." >&2
              echo "Run: git submodule update --init --recursive" >&2
              exit 1
            fi

            export UV_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/uv"
            export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
            export WARP_CACHE_PATH="$PWD/.warp-cache"
            export XLA_PYTHON_CLIENT_PREALLOCATE="''${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
            export TF_GPU_ALLOCATOR="''${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
            export MUJOCO_GL=${mujocoGl}
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
            ''}

            uv python install
            uv sync --frozen

            if [ -f .venv/bin/activate ]; then
              source .venv/bin/activate
            fi
          '';
        };
      }
    );
}
