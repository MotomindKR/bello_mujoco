#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )) || [[ ${1:-} != "" && ${1:-} != --arm && ${1:-} != --help ]]; then
  echo "Usage: bello-run-policy [--arm]" >&2
  exit 2
fi
if [[ ${1:-} == --help ]]; then
  echo "Usage: bello-run-policy [--arm]"
  echo "No arguments: read-only preflight. --arm: interactive HOLD/RUN/IDLE control."
  exit 0
fi

source_dir="$HOME/bello_mujoco"
policy="$source_dir/outputs/bello-mini-refactor-update-061440-2973427.zip"
config="$HOME/.config/bello/jetson-policy.yaml"
binary="$HOME/.cache/bello-mujoco/rust-runtime/release/bello-policy"
for file in "$policy" "$config" "$binary"; do
  if [[ ! -f "$file" ]]; then
    echo "Required deployment file missing: $file" >&2
    exit 1
  fi
done

cd "$source_dir"
if [[ ${1:-} == --arm ]]; then
  for unit in bello-hardwared bello-state-estimatord bello-reference bello-hands; do
    if ! systemctl --user is-active --quiet "$unit"; then
      echo "Required daemon is not running: $unit. Check systemctl --user status $unit." >&2
      exit 1
    fi
  done
  if pgrep -x bello-policy >/dev/null; then
    echo "A policy runner is already running; stop it through its normal IDLE sequence." >&2
    exit 1
  fi
  mkdir -p "$source_dir/outputs/policy_logs"
  session_dir=$(mktemp -d "$source_dir/outputs/policy_logs/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
  cd "$session_dir"
  echo "Policy run directory: $session_dir"
fi

exec /nix/var/nix/profiles/default/bin/nix develop "$source_dir#rust-runtime" --command \
  "$binary" --policy "$policy" --source "$source_dir" --hardware-config "$config" "$@"
