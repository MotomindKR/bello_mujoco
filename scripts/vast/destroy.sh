#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."
source scripts/vast/common.sh

INSTANCE_ID=""
YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --yes|-y) YES=1; shift ;;
    --help|-h)
      echo "Usage: scripts/vast/destroy.sh [--instance-id ID] [--yes]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

INSTANCE_ID=$(instance_id_or_die "$INSTANCE_ID")
hydrate_vast_cli_key

if [ "$YES" -ne 1 ]; then
  read -r -p "Destroy Vast instance $INSTANCE_ID and remove all remote data? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "aborted"; exit 0 ;;
  esac
fi

printf 'y\n' | vastai_cmd destroy instance "$INSTANCE_ID"
rm -f "$VAST_INSTANCE_FILE"
