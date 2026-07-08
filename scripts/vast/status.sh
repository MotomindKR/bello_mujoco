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
    --session) VAST_TMUX_SESSION="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: scripts/vast/status.sh [--instance-id ID] [--tail N] [--repo-dir PATH] [--session NAME]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

INSTANCE_ID=$(instance_id_or_die "$INSTANCE_ID")

info "Vast instance:"
vastai_cmd show instance "$INSTANCE_ID" || true

info "SSH URL:"
vastai_cmd ssh-url "$INSTANCE_ID" || true

info "remote training status:"
remote_ssh "$INSTANCE_ID" "bash -lc $(printf '%q' "
set -Eeuo pipefail
cd '$VAST_REMOTE_DIR' 2>/dev/null || { echo 'repo dir missing: $VAST_REMOTE_DIR'; exit 0; }
if tmux has-session -t '$VAST_TMUX_SESSION' 2>/dev/null; then
  echo 'tmux session: running'
  tmux capture-pane -pt '$VAST_TMUX_SESSION' -S -'$TAIL_LINES' || true
else
  echo 'tmux session: not running'
  ls -1t outputs/vast/*.log 2>/dev/null | head -1 | xargs -r tail -n '$TAIL_LINES'
fi
")"
