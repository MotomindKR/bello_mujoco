#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

if [ "$DRY_RUN" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  run apt-get update
  run apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    python3 \
    tmux
fi

if ! command -v uv >/dev/null 2>&1; then
  run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ "$DRY_RUN" -eq 0 ] && ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found after installation" >&2
  exit 1
fi

source scripts/vast/remote_env.sh

run uv python install 3.12
run uv sync
run uv run python -c 'import jax, mujoco, wandb; print({"backend": jax.default_backend(), "jax": jax.__version__, "mujoco": mujoco.__version__, "wandb": wandb.__version__})'
run uv run bello-train-ppo dry_run=true config.num_envs=1 config.num_steps=1 config.num_minibatches=1
