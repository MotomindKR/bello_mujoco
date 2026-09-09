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
        limaRuntimeLibraries = with pkgs; [
          libglvnd
          stdenv.cc.cc.lib
          udev
        ];
      in
      {
        devShells = {
          default = pkgs.mkShell {
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
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          lima-vcan = pkgs.mkShell {
            packages = with pkgs; [
              buf
              can-utils
              cargo
              clippy
              git
              grpcurl
              iproute2
              kmod
              openssl
              pkg-config
              protobuf
              rustc
              rustfmt
              socat
              uv
            ] ++ limaRuntimeLibraries;

            RUST_BACKTRACE = "1";

            shellHook = ''
              if [ ! -f deps/bello-stack/components/bello-hardware-python/pyproject.toml ]; then
                echo "The pinned bello-stack submodule is not initialized." >&2
                echo "Run on the Mac: git submodule update --init --recursive" >&2
                exit 1
              fi

              lima_state="''${XDG_STATE_HOME:-$HOME/.local/state}/bello-mujoco/lima-vcan"
              lima_cache="''${XDG_CACHE_HOME:-$HOME/.cache}/bello-mujoco/lima-vcan"
              export UV_PROJECT_ENVIRONMENT="$lima_state/venv"
              export UV_PYTHON_INSTALL_DIR="$lima_state/python"
              export UV_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/uv"
              export WARP_CACHE_PATH="$lima_cache/warp"
              export CARGO_TARGET_DIR="$lima_cache/cargo-target"
              export PYTHONPYCACHEPREFIX="$lima_cache/pycache"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath limaRuntimeLibraries}:''${LD_LIBRARY_PATH:-}"
              export JAX_PLATFORMS=cpu
              export XLA_PYTHON_CLIENT_PREALLOCATE=false
              export MUJOCO_GL=egl

              mkdir -p "$lima_state" "$lima_cache"
              uv python install
              uv sync --frozen

              if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
                source "$UV_PROJECT_ENVIRONMENT/bin/activate"
              fi
            '';
          };
        };
      }
    );
}
