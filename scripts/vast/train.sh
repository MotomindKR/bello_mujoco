#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."
source scripts/vast/common.sh

usage() {
  cat <<'EOF'
Usage: scripts/vast/train.sh [options] [-- HYDRA_OVERRIDES...]

Start PPO training on the Vast instance inside tmux.
If WANDB_API_KEY is unset, the script tries `keyring get wandb api_key`.
By default, the instance destroys itself after the training command exits.

Options:
  --instance-id ID      Vast instance id. Defaults to .vast/instance_id.
  --config-name NAME    Hydra train config. Default: vast_smoke.
  --session NAME        tmux session. Default: bello-train.
  --repo-dir PATH       Remote repo dir. Default: /workspace/bello_mujoco.
  --run-name NAME       W&B/checkpoint run name override.
  --keep-instance       Do not destroy the Vast instance when training exits.
  --help                Show this help.

Examples:
  scripts/vast/train.sh
  scripts/vast/train.sh -- config.total_timesteps=1048576
EOF
}

INSTANCE_ID=""
RUN_NAME=""
DESTROY_ON_EXIT=1
OVERRIDES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --config-name) VAST_TRAIN_CONFIG="$2"; shift 2 ;;
    --session) VAST_TMUX_SESSION="$2"; shift 2 ;;
    --repo-dir) VAST_REMOTE_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --keep-instance) DESTROY_ON_EXIT=0; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) OVERRIDES+=("$1"); shift ;;
  esac
done

INSTANCE_ID=$(instance_id_or_die "$INSTANCE_ID")
wait_for_ssh "$INSTANCE_ID" 600 || die "SSH is not ready for instance $INSTANCE_ID; check with scripts/vast/status.sh and retry"

if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY="$(secret_from_keyring wandb api_key)"
fi
[ -n "${WANDB_API_KEY:-}" ] || die "WANDB_API_KEY is required; export it or store it with: keyring set wandb api_key"

WANDB_PROJECT="${WANDB_PROJECT:-bello-mujoco}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_NAME="${RUN_NAME:-vast_smoke_seed0}"
DESTROY_CMD=":"
if [ "$DESTROY_ON_EXIT" -eq 1 ]; then
  VAST_DESTROY_API_KEY="$(vast_api_key)"
  [ -n "$VAST_DESTROY_API_KEY" ] || die "Vast API key is required for default destroy-on-exit; run 'vastai set api-key', export VAST_API_KEY, or pass --keep-instance"
  printf -v VAST_DESTROY_API_KEY_Q '%q' "$VAST_DESTROY_API_KEY"
  DESTROY_CMD="printf 'y\n' | uvx --from vastai vastai destroy instance '$INSTANCE_ID' --api-key $VAST_DESTROY_API_KEY_Q || true"
fi

REMOTE_OVERRIDES=(
  "wandb_config.enabled=true"
  "wandb_config.project=$WANDB_PROJECT"
  "wandb_config.run_name=$RUN_NAME"
  "checkpoint_config.name=$RUN_NAME"
  "checkpoint_config.wandb_artifact_name=$RUN_NAME"
  "${OVERRIDES[@]}"
)
if [ -n "$WANDB_ENTITY" ]; then
  REMOTE_OVERRIDES+=("wandb_config.entity=$WANDB_ENTITY")
fi
printf -v OVERRIDE_TEXT '%q ' "${REMOTE_OVERRIDES[@]}"

REMOTE_CMD=$(cat <<EOF
set -Eeuo pipefail
cd "$VAST_REMOTE_DIR"
mkdir -p outputs/vast
tmux kill-session -t "$VAST_TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$VAST_TMUX_SESSION" 'bash -lc '"'"'
set -Eeuo pipefail
cd "$VAST_REMOTE_DIR"
source scripts/vast/remote_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_ENTITY="$WANDB_ENTITY"
set +e
uv run bello-train-ppo --config-name "$VAST_TRAIN_CONFIG" $OVERRIDE_TEXT 2>&1 | tee "outputs/vast/$RUN_NAME.log"
train_status=\${PIPESTATUS[0]}
set -e
$DESTROY_CMD
exit "\$train_status"
'"'"''
tmux ls
EOF
)

remote_ssh "$INSTANCE_ID" "bash -lc $(printf '%q' "$REMOTE_CMD")"

cat <<EOF
Training started in tmux session '$VAST_TMUX_SESSION' on instance $INSTANCE_ID.
Destroy on exit: $DESTROY_ON_EXIT

Monitor:
  scripts/vast/status.sh
EOF
