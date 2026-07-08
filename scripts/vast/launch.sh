#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."
source scripts/vast/common.sh

usage() {
  cat <<'EOF'
Usage: scripts/vast/launch.sh [options]

Search and rent a Vast.ai RTX 5090 instance, then bootstrap this repo remotely.

Options:
  --dry-run             Print commands without searching or renting.
  --offer-id ID         Use a specific offer id instead of searching.
  --repo-ref REF        Git ref to check out remotely. Default: main.
  --repo-url URL        Git repository URL. Default: MotomindKR/bello_mujoco.
  --image IMAGE         Docker image. Default: NVIDIA CUDA Ubuntu devel.
  --disk GB            Disk size in GB. Default: 80.
  --ram-gb GB           Minimum CPU RAM in GB. Default: 64.
  --cuda-min VERSION    Minimum host CUDA capability. Default: 13.0.
  --gpu-name NAME       Vast gpu_name filter. Default: RTX_5090.
  --help                Show this help.

Environment:
  VAST_API_KEY may be set, or run `vastai set api-key` beforehand.
EOF
}

DRY_RUN=0
OFFER_ID="${VAST_OFFER_ID:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --offer-id) OFFER_ID="$2"; shift 2 ;;
    --repo-ref) VAST_REPO_REF="$2"; shift 2 ;;
    --repo-url) VAST_REPO_URL="$2"; shift 2 ;;
    --image) VAST_IMAGE="$2"; shift 2 ;;
    --disk) VAST_DISK_GB="$2"; shift 2 ;;
    --ram-gb) VAST_CPU_RAM_GB="$2"; shift 2 ;;
    --cuda-min) VAST_CUDA_VERSION_MIN="$2"; shift 2 ;;
    --gpu-name) VAST_GPU_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

ensure_state_dir

QUERY="gpu_name=${VAST_GPU_NAME} num_gpus=1 verified=true direct_port_count>=1 rentable=true cpu_ram>=${VAST_CPU_RAM_GB} cuda_vers>=${VAST_CUDA_VERSION_MIN}"
ONSTART_CMD=$(cat <<EOF
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git tmux python3
mkdir -p "$(dirname "$VAST_REMOTE_DIR")"
if [ ! -d "$VAST_REMOTE_DIR/.git" ]; then
  git clone "$VAST_REPO_URL" "$VAST_REMOTE_DIR"
fi
cd "$VAST_REMOTE_DIR"
git fetch --all --prune
git checkout "$VAST_REPO_REF"
scripts/vast/remote_setup.sh
EOF
)

if [ "$DRY_RUN" -eq 1 ]; then
  cat <<EOF
Search command:
  vastai search offers '$QUERY' -o 'dlperf_usd-' --raw

Create command:
  vastai create instance ${OFFER_ID:-OFFER_ID_FROM_SEARCH} --image '$VAST_IMAGE' --disk '$VAST_DISK_GB' --onstart-cmd '<bootstrap>' --ssh --direct

Remote repo:
  $VAST_REPO_URL @ $VAST_REPO_REF -> $VAST_REMOTE_DIR
EOF
  exit 0
fi

hydrate_vast_cli_key

if [ -z "$OFFER_ID" ]; then
  info "searching Vast offers: $QUERY"
  OFFER_ID=$(vastai_cmd search offers "$QUERY" -o 'dlperf_usd-' --raw | json_first_offer_id)
fi

info "creating instance from offer $OFFER_ID"
INSTANCE_ID=$(
  vastai_cmd create instance "$OFFER_ID" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK_GB" \
    --onstart-cmd "$ONSTART_CMD" \
    --ssh \
    --direct \
    --raw | json_new_instance_id
)
printf '%s\n' "$INSTANCE_ID" > "$VAST_INSTANCE_FILE"
info "created instance $INSTANCE_ID; saved to $VAST_INSTANCE_FILE"

RUNNING=0
for _ in $(seq 1 60); do
  STATUS=$(vastai_cmd show instance "$INSTANCE_ID" --raw | json_instance_status || true)
  info "instance $INSTANCE_ID status: $STATUS"
  case "$STATUS" in
    running)
      RUNNING=1
      break
      ;;
    exited|offline|unknown)
      die "instance entered non-recoverable status: $STATUS"
      ;;
  esac
  sleep 10
done

[ "$RUNNING" -eq 1 ] || die "instance did not reach running status within 10 minutes"
wait_for_ssh "$INSTANCE_ID" 600 || die "SSH did not become ready within 10 minutes"

info "ssh url:"
vastai_cmd ssh-url "$INSTANCE_ID"
cat <<EOF

Next:
  scripts/vast/status.sh
  WANDB_API_KEY=... scripts/vast/train.sh
  scripts/vast/destroy.sh
EOF
