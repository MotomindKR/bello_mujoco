#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help && $# == 1 ]]; then
  echo "Usage: bello-run-visualizer (run in an Ubuntu desktop terminal)"
  exit 0
fi
if (( $# != 0 )); then
  echo "Usage: bello-run-visualizer" >&2
  exit 2
fi
if [[ -z ${DISPLAY:-} && -z ${WAYLAND_DISPLAY:-} ]]; then
  echo "Run this launcher in an Ubuntu desktop terminal, not a headless SSH session." >&2
  exit 1
fi

config="$HOME/.config/bello/ubuntu-visualizer.yaml"
if [[ ! -f "$config" ]]; then
  echo "Required deployment file missing: $config" >&2
  exit 1
fi
cd "$HOME/bello_mujoco/deps/bello-stack"
exec /nix/var/nix/profiles/default/bin/nix develop --command \
  cargo run --locked --release -p bello-visualizer -- --config "$config"
