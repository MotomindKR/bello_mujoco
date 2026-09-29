#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: run_policy_service.sh BINARY POLICY SOURCE CONFIG" >&2
  exit 2
fi
: "${LOGS_DIRECTORY:?systemd must provide LogsDirectory}"
session_dir=$(mktemp -d "$LOGS_DIRECTORY/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
cd "$session_dir"
echo "Policy run directory: $session_dir"
exec "$1" --policy "$2" --source "$3" --hardware-config "$4" --arm
