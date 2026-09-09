#!/usr/bin/env bash
# Manage the macOS Lima guest used for Bello vCAN policy tests.
set -euo pipefail

readonly script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly repo_dir=$(cd -- "$script_dir/.." && pwd -P)
readonly template_path="$repo_dir/lima/bello-vcan.yaml"
readonly guest_repo=/workspace/bello_mujoco
readonly nix_bin=/nix/var/nix/profiles/default/bin/nix
readonly instance=${BELLO_LIMA_INSTANCE:-bello-vcan}
readonly environment_token=bello-mujoco-vcan-v1
readonly action=${1:-help}

usage() {
  cat <<'EOF'
Usage: ./scripts/lima_vcan.sh <command> [arguments]

Commands:
  start       Create or start the Lima VM and prepare vcan0/vcan1 and IMU PTYs.
  bootstrap   Build the Linux Nix shell and synchronize its isolated uv environment.
  shell       Open an interactive Linux Nix development shell.
  run CMD...  Run CMD inside the Linux Nix shell (TTY is preserved).
  doctor      Verify the VM architecture, Nix shell, vCAN links, IMU PTYs, and JAX backend.
  status      Show the Lima, vCAN, and IMU endpoint status.
  stop        Stop the IMU PTYs and the Lima VM; vCAN disappears with the VM.
  delete      Stop and permanently delete this Lima VM.

Set BELLO_LIMA_INSTANCE to use a different instance name.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_host() {
  [[ $(uname -s) == Darwin ]] || fail "this helper must run on macOS"
  command -v limactl >/dev/null 2>&1 \
    || fail "Lima is unavailable; install it with: brew install lima"
  [[ $instance =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "BELLO_LIMA_INSTANCE contains unsupported characters: $instance"
}

require_standalone_checkout() {
  local submodule_status
  if [[ ! -d $repo_dir/.git ]]; then
    fail "use this workflow from a standalone Git clone, not a linked worktree"
  fi
  submodule_status=$(git -C "$repo_dir" submodule status --recursive) \
    || fail "cannot inspect recursive submodule status"
  if grep -Eq '^[^ ]' <<<"$submodule_status"; then
    fail "submodules are not at their initialized pinned commits; run: git submodule update --init --recursive"
  fi
}

instance_exists() {
  limactl list --format '{{.Name}}' 2>/dev/null | grep -Fxq "$instance"
}

instance_status() {
  limactl list "$instance" --format '{{.Status}}' 2>/dev/null
}

instance_environment() {
  limactl list "$instance" \
    --format '{{index .Config.Param "ENVIRONMENT"}}' 2>/dev/null
}

instance_repo_mount() {
  limactl list "$instance" \
    --format '{{(index .Config.Mounts 0).Location}}|{{(index .Config.Mounts 0).MountPoint}}' \
    2>/dev/null
}

require_owned_instance() {
  local configured_environment
  configured_environment=$(instance_environment) \
    || fail "cannot inspect Lima instance $instance"
  [[ $configured_environment == "$environment_token" ]] || fail \
    "refusing to operate on unrelated Lima instance $instance"
}

checkout_matches_instance() {
  [[ $(instance_repo_mount) == "$repo_dir|$guest_repo" ]]
}

require_matching_checkout() {
  local mounted_repo
  mounted_repo=$(instance_repo_mount) \
    || fail "cannot inspect the repository mount for Lima instance $instance"
  [[ $mounted_repo == "$repo_dir|$guest_repo" ]] || fail \
    "Lima instance $instance has a different repository mount ($mounted_repo); delete it and run: $0 start"
}

require_running_instance() {
  instance_exists || fail "Lima instance $instance does not exist; run: $0 start"
  require_owned_instance
  [[ $(instance_status) == Running ]] \
    || fail "Lima instance $instance is not running; run: $0 start"
  require_matching_checkout
}

guest_shell() {
  limactl shell --workdir "$guest_repo" "$instance" "$@"
}

setup_emulator() {
  guest_shell sudo -- ./scripts/setup_emulator.sh "$1"
}

run_in_nix() {
  guest_shell "$nix_bin" develop .#lima-vcan --command "$@"
}

start_instance() {
  require_standalone_checkout
  if instance_exists; then
    require_owned_instance
    require_matching_checkout
    if [[ $(instance_status) == Running ]]; then
      echo "Lima instance $instance is already running."
    else
      limactl start "$instance"
    fi
  else
    limactl start --name "$instance" --param "REPO_DIR=$repo_dir" "$template_path"
  fi
  setup_emulator start
}

stop_instance() {
  if ! instance_exists; then
    echo "Lima instance $instance does not exist."
    return
  fi
  require_owned_instance
  if [[ $(instance_status) == Running ]]; then
    if checkout_matches_instance; then
      setup_emulator stop
    else
      echo "Repository mount differs from this checkout; skipping IMU PTY cleanup."
    fi
    limactl stop "$instance"
  else
    echo "Lima instance $instance is already stopped."
  fi
}

case $action in
  help|-h|--help)
    usage
    ;;
  start)
    (( $# == 1 )) || fail "start takes no arguments"
    require_host
    start_instance
    ;;
  bootstrap)
    (( $# == 1 )) || fail "bootstrap takes no arguments"
    require_host
    require_standalone_checkout
    require_running_instance
    setup_emulator status
    run_in_nix true
    echo "Bello Lima/Nix environment is synchronized."
    ;;
  shell)
    (( $# == 1 )) || fail "shell takes no arguments"
    require_host
    require_standalone_checkout
    require_running_instance
    guest_shell "$nix_bin" develop .#lima-vcan
    ;;
  run)
    (( $# >= 2 )) || fail "run requires a command"
    require_host
    require_standalone_checkout
    require_running_instance
    shift
    run_in_nix "$@"
    ;;
  doctor)
    (( $# == 1 )) || fail "doctor takes no arguments"
    require_host
    require_standalone_checkout
    require_running_instance
    echo "Lima instance: $instance ($(instance_status))"
    guest_shell uname -a
    guest_shell "$nix_bin" --version
    setup_emulator status
    run_in_nix python -c \
      'import jax, platform; print(f"Python platform: {platform.platform()}"); print(f"JAX backend: {jax.default_backend()}")'
    ;;
  status)
    (( $# == 1 )) || fail "status takes no arguments"
    require_host
    if ! instance_exists; then
      echo "Lima instance $instance does not exist."
      exit 1
    fi
    require_owned_instance
    limactl list "$instance"
    if [[ $(instance_status) == Running ]]; then
      if checkout_matches_instance; then
        setup_emulator status
      else
        echo "Repository mount differs from this checkout; guest emulator status skipped."
      fi
    fi
    ;;
  stop)
    (( $# == 1 )) || fail "stop takes no arguments"
    require_host
    stop_instance
    ;;
  delete)
    (( $# == 1 )) || fail "delete takes no arguments"
    require_host
    stop_instance
    if instance_exists; then
      limactl delete "$instance"
    fi
    ;;
  *)
    usage >&2
    fail "unknown command: $action"
    ;;
esac
