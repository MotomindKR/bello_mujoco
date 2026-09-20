#!/usr/bin/env bash
set -euo pipefail

motion=""
if [[ ${1:-} == --motion ]]; then
  case ${2:-} in
    randb|squat|lunge|balance) motion="$2"; shift 2 ;;
    *) echo "Choose a motion: randb, squat, lunge, balance." >&2; exit 2 ;;
  esac
fi
if (( $# > 1 )) || [[ ${1:-} != "" && ${1:-} != --arm && ${1:-} != --help ]]; then
  echo "Usage: bello-run-policy [--motion NAME] [--arm]" >&2
  exit 2
fi
if [[ ${1:-} == --help ]]; then
  echo "Usage: bello-run-policy [--motion NAME] [--arm]"
  echo "No arguments: read-only preflight. --arm: interactive HOLD/RUN/IDLE control."
  exit 0
fi

source_dir="$HOME/bello_mujoco"
policy="$source_dir/outputs/policies/locomotion.zip"
config="$HOME/.config/bello/jetson-policy.yaml"
if [[ -n "$motion" ]]; then
  policy="$source_dir/outputs/policies/$motion.zip"
  config="$source_dir/outputs/policies/configs/$motion.yaml"
fi
binary="$HOME/.cache/bello-mujoco/rust-runtime/release/bello-policy"
for file in "$policy" "$config" "$binary"; do
  if [[ ! -f "$file" ]]; then
    echo "Required deployment file missing: $file" >&2
    exit 1
  fi
done

if [[ -n "$motion" ]]; then
  env -u LD_LIBRARY_PATH "$source_dir/.venv/bin/python" - "$config" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as stream:
    config = yaml.safe_load(stream)
if config["control"]["max_duration_seconds"] is not None:
    raise SystemExit("Motion launcher requires control.max_duration_seconds: null (no run-duration limit).")
PY
fi

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
  session_dir=$(mktemp -d "$source_dir/outputs/policy_logs/${motion:-locomotion}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
  cd "$session_dir"
  echo "Policy run directory: $session_dir"
fi

exec /nix/var/nix/profiles/default/bin/nix develop "$source_dir#rust-runtime" --command \
  "$binary" --policy "$policy" --source "$source_dir" --hardware-config "$config" "$@"
