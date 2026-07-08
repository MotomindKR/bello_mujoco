#!/usr/bin/env bash
set -Eeuo pipefail

VAST_STATE_DIR="${VAST_STATE_DIR:-.vast}"
VAST_INSTANCE_FILE="${VAST_INSTANCE_FILE:-$VAST_STATE_DIR/instance_id}"
VAST_REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/bello_mujoco}"
VAST_REPO_URL="${VAST_REPO_URL:-https://github.com/MotomindKR/bello_mujoco.git}"
VAST_REPO_REF="${VAST_REPO_REF:-main}"
VAST_IMAGE="${VAST_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
VAST_DISK_GB="${VAST_DISK_GB:-80}"
VAST_GPU_NAME="${VAST_GPU_NAME:-RTX_5090}"
VAST_CPU_RAM_GB="${VAST_CPU_RAM_GB:-64}"
VAST_CUDA_VERSION_MIN="${VAST_CUDA_VERSION_MIN:-13.0}"
VAST_TMUX_SESSION="${VAST_TMUX_SESSION:-bello-train}"
VAST_TRAIN_CONFIG="${VAST_TRAIN_CONFIG:-vast_rtx5090}"

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "==> $*" >&2
}

ensure_state_dir() {
  mkdir -p "$VAST_STATE_DIR"
}

vastai_cmd() {
  if command -v vastai >/dev/null 2>&1; then
    vastai "$@"
  elif command -v uvx >/dev/null 2>&1; then
    uvx --from vastai vastai "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv tool run --from vastai vastai "$@"
  else
    die "vastai is not installed and uv/uvx is unavailable. Install uv or run: pipx install vastai"
  fi
}

instance_id_or_die() {
  local instance_id="${1:-}"
  if [ -n "$instance_id" ]; then
    printf '%s\n' "$instance_id"
    return
  fi
  [ -f "$VAST_INSTANCE_FILE" ] || die "no instance id supplied and $VAST_INSTANCE_FILE does not exist"
  tr -d '[:space:]' < "$VAST_INSTANCE_FILE"
}

ssh_target() {
  local instance_id="$1"
  vastai_cmd ssh-url "$instance_id" | tail -n 1 | python3 -c '
from urllib.parse import urlparse
import sys

text = sys.stdin.read().strip()
if not text.startswith("ssh://"):
    raise SystemExit(f"unexpected ssh-url output: {text!r}")
url = urlparse(text)
if not url.hostname or not url.port or not url.username:
    raise SystemExit(f"incomplete ssh-url output: {text!r}")
print(f"{url.username}@{url.hostname} {url.port}")
'
}

remote_ssh() {
  local instance_id="$1"
  shift
  local target port
  read -r target port < <(ssh_target "$instance_id")
  ssh \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ConnectTimeout=20 \
    -p "$port" \
    "$target" \
    "$@"
}

json_first_offer_id() {
  python3 -c '
import json
import sys

data = json.load(sys.stdin)
if isinstance(data, dict):
    offers = data.get("offers") or data.get("results") or data.get("data") or []
else:
    offers = data
if not offers:
    raise SystemExit("no offers matched query")
offer = offers[0]
print(offer.get("id") or offer.get("ask_contract_id") or offer.get("contract_id"))
'
}

json_new_instance_id() {
  python3 -c '
import json
import sys

data = json.load(sys.stdin)
for key in ("new_contract", "id", "instance_id", "contract_id"):
    if key in data:
        print(data[key])
        break
else:
    raise SystemExit(f"could not find instance id in create output: {data!r}")
'
}

json_instance_status() {
  python3 -c '
import json
import sys

data = json.load(sys.stdin)
if isinstance(data, list):
    data = data[0] if data else {}
print(data.get("actual_status") or data.get("status") or data.get("intended_status") or "unknown")
'
}
