#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help && $# == 1 ]]; then
  echo "Usage: bello-run-motion {randb|squat|lunge|balance} [--arm]"
  echo "Default: read-only preflight. --arm: interactive HOLD/RUN/MOTION/IDLE."
  echo "No run-duration limit. Each clip returns to standing; MOTION replays it."
  exit 0
fi
case ${1:-} in
  randb|squat|lunge|balance) ;;
  *) echo "Usage: bello-run-motion {randb|squat|lunge|balance} [--arm]" >&2; exit 2 ;;
esac
motion="$1"
shift
exec "$HOME/.local/bin/bello-run-policy" --motion "$motion" "$@"
