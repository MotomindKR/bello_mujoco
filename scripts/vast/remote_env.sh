#!/usr/bin/env bash
set -Eeuo pipefail

export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export WARP_CACHE_PATH="$PWD/.warp-cache"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=true}"
