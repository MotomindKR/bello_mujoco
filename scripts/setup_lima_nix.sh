#!/usr/bin/env bash
# Install and configure multi-user Nix inside the Bello Lima guest.
set -euo pipefail

if (( EUID != 0 )); then
  echo "error: setup_lima_nix.sh must run as root" >&2
  exit 1
fi

export HOME=/root
export USER=root

readonly nix_bin=/nix/var/nix/profiles/default/bin/nix
if [[ ! -x $nix_bin ]]; then
  readonly installer_dir=$(mktemp -d /tmp/bello-nix-installer.XXXXXX)
  readonly installer_path=$installer_dir/install
  cleanup() {
    rm -rf -- "$installer_dir"
  }
  trap cleanup EXIT

  echo "Nix is missing; downloading the multi-user installer."
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
    --location --retry 5 --retry-delay 2 --retry-all-errors \
    --connect-timeout 20 --output "$installer_path" \
    https://nixos.org/nix/install
  sh "$installer_path" --daemon --yes
fi

readonly nix_conf=/etc/nix/nix.conf
install -d --mode=0755 /etc/nix
touch "$nix_conf"
if ! grep -Eq '^[[:space:]]*experimental-features[[:space:]]*=.*\bflakes\b' \
  "$nix_conf"; then
  printf '\nexperimental-features = nix-command flakes\n' >> "$nix_conf"
fi
systemctl enable --now nix-daemon.socket
"$nix_bin" --version
