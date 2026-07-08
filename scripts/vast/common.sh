#!/usr/bin/env bash
set -Eeuo pipefail

VAST_STATE_DIR="${VAST_STATE_DIR:-.vast}"
VAST_INSTANCE_FILE="${VAST_INSTANCE_FILE:-$VAST_STATE_DIR/instance_id}"
VAST_REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/bello_mujoco}"
VAST_REPO_URL="${VAST_REPO_URL:-https://github.com/MotomindKR/bello_mujoco.git}"
VAST_REPO_REF="${VAST_REPO_REF:-test/vast-short-training}"
VAST_IMAGE="${VAST_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
VAST_DISK_GB="${VAST_DISK_GB:-80}"
VAST_GPU_NAME="${VAST_GPU_NAME:-RTX_5090}"
VAST_CPU_RAM_GB="${VAST_CPU_RAM_GB:-64}"
VAST_CUDA_VERSION_MIN="${VAST_CUDA_VERSION_MIN:-13.0}"
VAST_SETUP_SESSION="${VAST_SETUP_SESSION:-bello-setup}"
VAST_TMUX_SESSION="${VAST_TMUX_SESSION:-bello-train}"
VAST_TRAIN_CONFIG="${VAST_TRAIN_CONFIG:-vast_smoke}"

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

keyring_cmd() {
  if command -v keyring >/dev/null 2>&1; then
    keyring "$@"
  elif command -v uvx >/dev/null 2>&1; then
    uvx --from keyring keyring "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv tool run --from keyring keyring "$@"
  else
    return 127
  fi
}

secret_from_keyring() {
  local service="$1"
  local username="$2"
  keyring_cmd get "$service" "$username" 2>/dev/null || true
}

vast_api_key() {
  if [ -n "${VAST_API_KEY:-}" ]; then
    printf '%s\n' "$VAST_API_KEY"
    return
  fi
  local keyring_key
  keyring_key="$(secret_from_keyring vast api_key)"
  if [ -n "$keyring_key" ]; then
    printf '%s\n' "$keyring_key"
    return
  fi
  local key_file="${VAST_API_KEY_FILE:-$HOME/.config/vastai/vast_api_key}"
  if [ -r "$key_file" ]; then
    tr -d '[:space:]' < "$key_file"
  fi
}

hydrate_vast_cli_key() {
  local key
  key="$(vast_api_key)"
  if [ -n "$key" ]; then
    vastai_cmd set api-key "$key" >/dev/null
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
    -o ConnectTimeout="${VAST_SSH_CONNECT_TIMEOUT:-20}" \
    -p "$port" \
    "$target" \
    "$@"
}

wait_for_ssh() {
  local instance_id="$1"
  local timeout_seconds="${2:-900}"
  local deadline=$((SECONDS + timeout_seconds))

  while [ "$SECONDS" -lt "$deadline" ]; do
    if VAST_SSH_CONNECT_TIMEOUT=10 remote_ssh "$instance_id" "true" >/dev/null 2>&1; then
      return 0
    fi
    info "waiting for SSH on instance $instance_id"
    sleep 10
  done

  return 1
}

confirm_or_die() {
  local prompt="$1"
  read -r -p "$prompt [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) die "aborted" ;;
  esac
}

require_vast_ssh_key() {
  local count
  count=$(vastai_cmd show ssh-keys --raw | python3 -c '
import json
import sys

data = json.load(sys.stdin)
print(len(data if isinstance(data, list) else data.get("ssh_keys", [])))
')
  if [ "$count" -lt 1 ]; then
    die "no Vast SSH keys are registered; run: vastai create ssh-key -y"
  fi
}

copy_clean_ref_to_instance() {
  local instance_id="$1"
  local repo_ref="$2"
  local remote_dir="$3"
  local commit short_commit target port exported_at
  local archive_dir archive_path archive_name remote_archive
  local remote_dir_q remote_archive_q repo_ref_q commit_q exported_at_q

  commit=$(git rev-parse --verify "${repo_ref}^{commit}") || die "git ref not found locally: $repo_ref"
  short_commit="${commit:0:12}"
  archive_dir=$(mktemp -d "${TMPDIR:-/tmp}/bello-vast-copy.XXXXXX")
  archive_name="bello_mujoco_${short_commit}.tar.gz"
  archive_path="$archive_dir/$archive_name"
  remote_archive="/tmp/$archive_name"

  info "creating clean source archive for $repo_ref ($short_commit)"
  git archive --format=tar.gz "$commit" > "$archive_path"
  du -h "$archive_path" >&2 || true

  read -r target port < <(ssh_target "$instance_id")
  info "uploading source archive with scp"
  if ! scp \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ConnectTimeout="${VAST_SSH_CONNECT_TIMEOUT:-20}" \
    -P "$port" \
    "$archive_path" \
    "$target:$remote_archive"; then
    rm -rf "$archive_dir"
    die "scp source archive failed"
  fi

  exported_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf -v remote_dir_q '%q' "$remote_dir"
  printf -v remote_archive_q '%q' "$remote_archive"
  printf -v repo_ref_q '%q' "$repo_ref"
  printf -v commit_q '%q' "$commit"
  printf -v exported_at_q '%q' "$exported_at"

  remote_ssh "$instance_id" "set -Eeuo pipefail; rm -rf $remote_dir_q; mkdir -p $remote_dir_q; tar -xzf $remote_archive_q -C $remote_dir_q; rm -f $remote_archive_q; { printf 'repo_ref=%s\n' $repo_ref_q; printf 'commit=%s\n' $commit_q; printf 'exported_at=%s\n' $exported_at_q; } > $remote_dir_q/.vast_source"
  rm -rf "$archive_dir"

  remote_ssh "$instance_id" "test -f $(printf '%q' "$remote_dir/scripts/vast/remote_setup.sh")" ||
    die "repo copy did not produce $remote_dir/scripts/vast/remote_setup.sh"
}

start_remote_setup() {
  local instance_id="$1"
  local remote_dir="$2"
  local setup_session="$3"
  local remote_dir_q setup_session_q

  printf -v remote_dir_q '%q' "$remote_dir"
  printf -v setup_session_q '%q' "$setup_session"

  remote_ssh "$instance_id" "bash -s -- $remote_dir_q $setup_session_q" <<'REMOTE'
set -Eeuo pipefail
remote_dir="$1"
setup_session="$2"
runner=/tmp/bello_remote_setup_runner.sh

cd "$remote_dir"
mkdir -p outputs/vast
tmux kill-session -t "$setup_session" 2>/dev/null || true
cat > "$runner" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$remote_dir" || exit 1
rm -f outputs/vast/setup.status
set +e
scripts/vast/remote_setup.sh 2>&1 | tee outputs/vast/setup.log
setup_status=\${PIPESTATUS[0]}
set -e
printf '%s\n' "\$setup_status" > outputs/vast/setup.status
exit "\$setup_status"
EOF
chmod +x "$runner"
tmux new-session -d -s "$setup_session" "bash $runner"
tmux ls
REMOTE
}

wait_for_remote_setup() {
  local instance_id="$1"
  local remote_dir="$2"
  local setup_session="$3"
  local timeout_seconds="${4:-1800}"
  local deadline=$((SECONDS + timeout_seconds))
  local status

  while [ "$SECONDS" -lt "$deadline" ]; do
    status=$(
      remote_ssh "$instance_id" "bash -lc $(printf '%q' "
cd '$remote_dir' 2>/dev/null || { echo missing-repo; exit 0; }
if tmux has-session -t '$setup_session' 2>/dev/null; then
  echo running
elif [ -f outputs/vast/setup.status ]; then
  printf 'exit:'
  cat outputs/vast/setup.status
else
  echo missing-status
fi
")" | tail -n 1
    )

    case "$status" in
      running)
        info "remote setup still running"
        ;;
      exit:0)
        info "remote setup completed"
        return 0
        ;;
      exit:*)
        echo "remote setup failed with status ${status#exit:}; tail follows:" >&2
        remote_ssh "$instance_id" "bash -lc $(printf '%q' "cd '$remote_dir' && tail -n 120 outputs/vast/setup.log 2>/dev/null || true")" >&2 || true
        return 1
        ;;
      missing-repo|missing-status)
        info "remote setup status: $status"
        ;;
      *)
        info "remote setup status: $status"
        ;;
    esac
    sleep 20
  done

  return 1
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

json_first_offer() {
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
print(json.dumps(offers[0]))
'
}

json_offer_id() {
  python3 -c '
import json
import sys

offer = json.load(sys.stdin)
print(offer.get("id") or offer.get("ask_contract_id") or offer.get("contract_id"))
'
}

json_offer_by_id() {
  local offer_id="$1"
  python3 -c '
import json
import sys

offer_id = sys.argv[1]
data = json.load(sys.stdin)
if isinstance(data, dict):
    offers = data.get("offers") or data.get("results") or data.get("data") or []
else:
    offers = data
for offer in offers:
    ids = [offer.get("id"), offer.get("ask_contract_id"), offer.get("contract_id")]
    if offer_id in {str(value) for value in ids if value is not None}:
        print(json.dumps(offer))
        break
else:
    raise SystemExit(f"offer id not found in search results: {offer_id}")
' "$offer_id"
}

json_offer_summary() {
  python3 -c '
import json
import sys

offer = json.load(sys.stdin)

def get(*keys, default="-"):
    for key in keys:
        value = offer.get(key)
        if value not in (None, ""):
            return value
    return default

def gb_from_mb(value):
    try:
        return f"{float(value) / 1024:.1f} GB"
    except Exception:
        return "-"

rows = [
    ("offer id", get("id", "ask_contract_id", "contract_id")),
    ("machine id", get("machine_id", "host_id")),
    ("location", get("geolocation")),
    ("gpu", "{}x {}".format(get("num_gpus"), get("gpu_name"))),
    ("gpu ram", gb_from_mb(get("gpu_ram", "gpu_total_ram", default=0))),
    ("cpu", "{} vCPUs / {}".format(get("cpu_cores_effective", "cpu_cores"), get("cpu_name"))),
    ("cpu ram", gb_from_mb(get("cpu_ram", default=0))),
    ("disk", "{} GB {}".format(get("disk_space"), get("disk_name"))),
    ("cuda", get("cuda_max_good")),
    ("driver", get("driver_version", "driver_vers")),
    ("reliability", get("reliability2", "reliability")),
    ("dlperf", get("dlperf")),
    ("dlperf/$", get("dlperf_per_dphtotal")),
    ("price", "${:.4f}/hr".format(float(get("dph_total", "discounted_dph_total", default=0)))),
    ("inet up/down", "{} / {} Mbps".format(get("inet_up"), get("inet_down"))),
    ("verification", get("verification")),
]
for key, value in rows:
    print(f"{key:>12}: {value}")
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
