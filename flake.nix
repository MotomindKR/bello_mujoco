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
          mesa
          stdenv.cc.cc.lib
          udev
        ];
        pythonShell = training: pkgs.mkShell {
          packages = [ pkgs.git pkgs.socat pkgs.uv ]
            ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.iproute2 pkgs.kmod ];
          UV_PROJECT_ENVIRONMENT = if training then ".venv-training" else ".venv";
          JAX_PLATFORMS = if training then "cuda" else "cpu";
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
            export MUJOCO_GL=${mujocoGl}
            ${pkgs.lib.optionalString training ''
              export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
              export TF_GPU_ALLOCATOR=cuda_malloc_async
            ''}
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath limaRuntimeLibraries}:''${LD_LIBRARY_PATH:-}"
              export __EGL_VENDOR_LIBRARY_FILENAMES=${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json
            ''}

            uv python install --no-bin || exit 1
            uv sync --frozen ${pkgs.lib.optionalString training "--group training"} || exit 1
            source "$UV_PROJECT_ENVIRONMENT/bin/activate"
          '';
        };
      in
      {
        devShells = {
          default = pythonShell false;
        } // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
          training = pythonShell true;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin {
          macos-visualizer = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              git
              protobuf
              rustc
              rustfmt
            ];

            RUST_BACKTRACE = "1";

            shellHook = ''
              if [ ! -f deps/bello-stack/components/bello-visualizer/Cargo.toml ]; then
                echo "The pinned bello-stack submodule is not initialized." >&2
                echo "Run: git submodule update --init --recursive" >&2
                exit 1
              fi

              macos_visualizer_cache="''${XDG_CACHE_HOME:-$HOME/.cache}/bello-mujoco/macos-visualizer"
              export CARGO_HOME="$macos_visualizer_cache/cargo-home"
              export CARGO_TARGET_DIR="$macos_visualizer_cache/cargo-target"
              mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"
            '';
          };
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          rust-runtime = pkgs.mkShell {
            packages = with pkgs; [ cargo rustc rustfmt clippy protobuf pkg-config mujoco onnxruntime ];
            MUJOCO_INCLUDE_DIR = "${pkgs.lib.getDev pkgs.mujoco}/include";
            MUJOCO_LIB_DIR = "${pkgs.lib.getLib pkgs.mujoco}/lib";
            LIBCLANG_PATH = "${pkgs.lib.getLib pkgs.llvmPackages.libclang}/lib";
            ORT_DYLIB_PATH = "${pkgs.lib.getLib pkgs.onnxruntime}/lib/libonnxruntime.so";
            shellHook = ''
              export CARGO_TARGET_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/bello-mujoco/rust-runtime"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.mujoco pkgs.onnxruntime ]}:''${LD_LIBRARY_PATH:-}"
            '';
          };
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
              export __EGL_VENDOR_LIBRARY_FILENAMES=${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json
              export JAX_PLATFORMS=cpu
              export XLA_PYTHON_CLIENT_PREALLOCATE=false
              export MUJOCO_GL=egl
              export NO_PROXY="host.lima.internal,127.0.0.1,localhost''${NO_PROXY:+,$NO_PROXY}"
              export no_proxy="$NO_PROXY"

              mkdir -p "$lima_state" "$lima_cache"
              uv python install --no-bin || exit 1
              uv sync --frozen || exit 1
              source "$UV_PROJECT_ENVIRONMENT/bin/activate"
            '';
          };
        };
      }
    );
}
