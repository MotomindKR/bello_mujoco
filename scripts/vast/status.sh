#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."
source scripts/vast/common.sh

INSTANCE_ID=""
TAIL_LINES="${TAIL_LINES:-120}"

while [ $# -gt 0 ]; do
  case "$1" in
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --tail) TAIL_LINES="$2"; shift 2 ;;
    --repo-dir) VAST_REMOTE_DIR="$2"; shift 2 ;;
    --setup-session) VAST_SETUP_SESSION="$2"; shift 2 ;;
    --session) VAST_TMUX_SESSION="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: scripts/vast/status.sh [--instance-id ID] [--tail N] [--repo-dir PATH] [--setup-session NAME] [--session NAME]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

INSTANCE_ID=$(instance_id_or_die "$INSTANCE_ID")
hydrate_vast_cli_key

info "Vast instance:"
vastai_cmd show instance "$INSTANCE_ID" || true

info "SSH URL:"
vastai_cmd ssh-url "$INSTANCE_ID" || true

info "remote training status:"
if ! remote_ssh "$INSTANCE_ID" "bash -lc $(printf '%q' "
set -Eeuo pipefail
cd '$VAST_REMOTE_DIR' 2>/dev/null || { echo 'repo dir missing: $VAST_REMOTE_DIR'; exit 0; }
echo 'source:'
cat .vast_source 2>/dev/null || echo 'source metadata missing'
echo
if tmux has-session -t '$VAST_SETUP_SESSION' 2>/dev/null; then
  echo 'setup tmux session: running'
  tmux capture-pane -pt '$VAST_SETUP_SESSION' -S -'$TAIL_LINES' || true
elif [ -f outputs/vast/setup.log ]; then
  echo 'setup tmux session: not running'
  tail -n '$TAIL_LINES' outputs/vast/setup.log || true
else
  echo 'setup log: missing'
fi
echo
if tmux has-session -t '$VAST_TMUX_SESSION' 2>/dev/null; then
  echo 'tmux session: running'
  tmux capture-pane -pt '$VAST_TMUX_SESSION' -S -'$TAIL_LINES' || true
else
  echo 'tmux session: not running'
  latest_log=\$(find outputs/vast -maxdepth 1 -type f -name '*.log' ! -name setup.log 2>/dev/null | sort | tail -1 || true)
  if [ -n \"\$latest_log\" ]; then
    tail -n '$TAIL_LINES' \"\$latest_log\" || true
  else
    echo 'training log: missing'
  fi
fi
exit 0
")"; then
  echo "SSH is not ready yet; wait a bit and rerun scripts/vast/status.sh"
fi
