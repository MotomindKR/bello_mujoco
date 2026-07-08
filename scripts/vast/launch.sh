#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."
source scripts/vast/common.sh

usage() {
  cat <<'EOF'
Usage: scripts/vast/launch.sh [options]

Search and rent a Vast.ai RTX 5090 instance, then bootstrap this repo remotely.
The instance downloads the requested GitHub ref over HTTPS using a local token
from GITHUB_TOKEN or keyring.

Options:
  --dry-run             Print commands without searching or renting.
  --offer-id ID         Use a specific offer id instead of searching.
  --yes                 Skip the selected-offer confirmation prompt.
  --no-wait-setup       Return after starting remote setup instead of waiting.
  --repo-ref REF        GitHub ref to download. Default: test/vast-short-training.
  --repo-url URL        GitHub repository URL. Default: https://github.com/MotomindKR/bello_mujoco.git.
  --image IMAGE         Docker image. Default: vastai/base-image:cuda-13.0.3-auto.
  --disk GB            Disk size in GB. Default: 80.
  --ram-gb GB           Minimum CPU RAM in GB. Default: 64.
  --cuda-min VERSION    Minimum host CUDA capability. Default: 13.0.
  --gpu-name NAME       Vast gpu_name filter. Default: RTX_5090.
  --help                Show this help.

Environment:
  VAST_API_KEY may be set, or run `vastai set api-key` beforehand.
  GITHUB_TOKEN may be set, or store one with `keyring set github bello_mujoco_read_token`.
EOF
}

DRY_RUN=0
YES="${VAST_YES:-0}"
WAIT_SETUP="${VAST_WAIT_SETUP:-1}"
OFFER_ID="${VAST_OFFER_ID:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --offer-id) OFFER_ID="$2"; shift 2 ;;
    --yes|-y) YES=1; shift ;;
    --no-wait-setup) WAIT_SETUP=0; shift ;;
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

if [ "$DRY_RUN" -eq 1 ]; then
  cat <<EOF
Search command:
  vastai search offers '$QUERY' -o 'dlperf_usd-' --raw

Create command:
  vastai create instance ${OFFER_ID:-OFFER_ID_FROM_SEARCH} --image '$VAST_IMAGE' --disk '$VAST_DISK_GB' --ssh --direct

Source download on instance:
  curl -L -H 'Authorization: Bearer <GITHUB_TOKEN>' '$(github_tarball_url "$VAST_REPO_URL" "$VAST_REPO_REF")' -o /tmp/bello_mujoco.tar.gz
  ssh root@INSTANCE 'tar -xzf /tmp/bello_mujoco.tar.gz --strip-components=1 -C $VAST_REMOTE_DIR'

Remote setup:
  tmux session '$VAST_SETUP_SESSION'
  log $VAST_REMOTE_DIR/outputs/vast/setup.log
  wait for completion: $WAIT_SETUP
EOF
  exit 0
fi

hydrate_vast_cli_key
require_vast_ssh_key
require_github_token

OFFER_JSON=""
if [ -z "$OFFER_ID" ]; then
  info "searching Vast offers: $QUERY"
  OFFERS_JSON=$(vastai_cmd search offers "$QUERY" -o 'dlperf_usd-' --raw)
  OFFER_JSON=$(printf '%s\n' "$OFFERS_JSON" | json_first_offer)
  OFFER_ID=$(printf '%s\n' "$OFFER_JSON" | json_offer_id)
else
  info "querying Vast offer metadata: $OFFER_ID"
  OFFERS_JSON=$(vastai_cmd search offers "ask_contract_id=$OFFER_ID" --raw 2>/dev/null)
  OFFER_JSON=$(printf '%s\n' "$OFFERS_JSON" | json_offer_by_id "$OFFER_ID")
fi

info "selected Vast offer:"
printf '%s\n' "$OFFER_JSON" | json_offer_summary
cat <<EOF

Launch settings:
       image: $VAST_IMAGE
        disk: $VAST_DISK_GB GB
    repo ref: $VAST_REPO_REF
    repo url: $VAST_REPO_URL
  remote dir: $VAST_REMOTE_DIR
 setup tmux: $VAST_SETUP_SESSION
EOF
if [ "$YES" -ne 1 ]; then
  confirm_or_die "Rent this Vast instance"
fi
printf '%s\n' "$OFFER_JSON" > "$VAST_STATE_DIR/offer_$OFFER_ID.json"

info "creating instance from offer $OFFER_ID"
INSTANCE_ID=$(
  vastai_cmd create instance "$OFFER_ID" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK_GB" \
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

info "downloading $VAST_REPO_REF from GitHub on instance $INSTANCE_ID"
download_github_ref_to_instance "$INSTANCE_ID" "$VAST_REPO_URL" "$VAST_REPO_REF" "$VAST_REMOTE_DIR"

info "starting remote setup in tmux session $VAST_SETUP_SESSION"
start_remote_setup "$INSTANCE_ID" "$VAST_REMOTE_DIR" "$VAST_SETUP_SESSION"
if [ "$WAIT_SETUP" -eq 1 ]; then
  wait_for_remote_setup "$INSTANCE_ID" "$VAST_REMOTE_DIR" "$VAST_SETUP_SESSION" 1800 || die "remote setup did not complete successfully; inspect with scripts/vast/status.sh"
else
  info "remote setup is running asynchronously; monitor with scripts/vast/status.sh"
fi

info "ssh url:"
vastai_cmd ssh-url "$INSTANCE_ID"
cat <<EOF

Next:
  scripts/vast/status.sh
  scripts/vast/train.sh
  scripts/vast/destroy.sh
EOF
