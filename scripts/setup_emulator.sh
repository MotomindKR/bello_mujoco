#!/usr/bin/env bash
# Prepare the host interfaces required by the Bello hardware emulator.
set -euo pipefail

readonly queue_length=10000
readonly interfaces=(vcan0 vcan1)
readonly action=${1:-start}

if (( $# > 1 )) || [[ $action != start && $action != stop && $action != status ]]; then
  echo "Usage: $0 [start|stop|status]" >&2
  exit 2
fi

socat_bin=${SOCAT_BIN:-}
if [[ $action == start && -z $socat_bin ]]; then
  socat_bin=$(command -v socat || true)
fi
if [[ $action == start && -z $socat_bin ]]; then
  echo "socat is unavailable; enter this repository's nix develop shell first." >&2
  exit 1
fi

if (( EUID != 0 )); then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "VCAN and emulated UART setup require root privileges, but sudo is unavailable." >&2
    exit 1
  fi
  exec sudo -- env "SOCAT_BIN=$socat_bin" "$0" "$@"
fi

required_commands=()
if [[ $action == start ]]; then
  required_commands=(ip modprobe)
elif [[ $action == status ]]; then
  required_commands=(ip)
fi
for command in "${required_commands[@]}"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command" >&2
    exit 1
  fi
done

with_serial_pair() {
  local profile=$1
  shift
  local host_device mujoco_device pid_file log_file label
  case $profile in
    imu) host_device=/dev/ttyIMU; label=EBIMU ;;
    power) host_device=/dev/ttyPOWER; label=COG-power ;;
    *) echo "Unknown serial profile: $profile" >&2; return 2 ;;
  esac
  mujoco_device="${host_device}-mujoco"
  pid_file="/run/bello-mujoco-${profile}-socat.pid"
  log_file="/tmp/bello-mujoco-${profile}-socat.log"
  "$@"
}

serial_running() {
  [[ -f $pid_file ]] || return 1
  local pid
  read -r pid < "$pid_file"
  [[ $pid =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r /proc/$pid/cmdline ]] || return 1
  local command_line
  command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  [[ $command_line == *socat* && $command_line == *"$host_device"* && $command_line == *"$mujoco_device"* ]]
}

remove_serial_endpoint() {
  local path=$1
  if [[ -e $path && ! -L $path ]]; then
    echo "Refusing to replace non-symlink device: $path" >&2
    exit 1
  fi
  rm -f -- "$path"
}

start_vcan() {
  modprobe vcan

  for interface in "${interfaces[@]}"; do
    if ip link show dev "$interface" >/dev/null 2>&1; then
      details=$(ip -details -json link show dev "$interface")
      if [[ $details != *'"info_kind":"vcan"'* ]]; then
        echo "Interface $interface already exists but is not a VCAN device." >&2
        exit 1
      fi
      echo "Using existing $interface"
    else
      ip link add dev "$interface" type vcan
      echo "Created $interface"
    fi

    ip link set dev "$interface" txqueuelen "$queue_length"
    ip link set dev "$interface" up
  done

  echo "VCAN interfaces are ready:"
  for interface in "${interfaces[@]}"; do
    ip -brief link show dev "$interface"
  done
}

start_serial() {
  if serial_running; then
    echo "$label serial pair is already running (PID $(<"$pid_file"))."
    echo "  driver: $host_device"
    echo "  MuJoCo: $mujoco_device"
    return
  fi

  rm -f -- "$pid_file"
  remove_serial_endpoint "$host_device"
  remove_serial_endpoint "$mujoco_device"
  : > "$log_file"
  nohup "$socat_bin" -d -d \
    "PTY,raw,echo=0,link=$host_device,mode=666" \
    "PTY,raw,echo=0,link=$mujoco_device,mode=666" \
    </dev/null >>"$log_file" 2>&1 &
  pid=$!
  echo "$pid" > "$pid_file"

  for _ in {1..50}; do
    if [[ -L $host_device && -L $mujoco_device ]]; then
      chmod 666 "$(readlink -f "$host_device")" "$(readlink -f "$mujoco_device")"
      echo "$label serial pair is ready (PID $pid):"
      echo "  Motomind driver endpoint: $host_device"
      echo "  MuJoCo emulator endpoint: $mujoco_device"
      return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done

  echo "socat did not create the $label serial endpoints:" >&2
  tail -20 "$log_file" >&2
  exit 1
}

stop_serial() {
  if serial_running; then
    pid=$(<"$pid_file")
    kill "$pid"
    wait "$pid" 2>/dev/null || true
    echo "Stopped $label serial pair (PID $pid)."
  else
    echo "$label serial pair is not running."
  fi
  rm -f -- "$pid_file"
  remove_serial_endpoint "$host_device"
  remove_serial_endpoint "$mujoco_device"
}

show_status() {
  local failed=0
  for interface in "${interfaces[@]}"; do
    if ! ip -brief link show dev "$interface"; then
      failed=1
    fi
  done
  for profile in imu power; do
    if ! with_serial_pair "$profile" show_serial_status; then failed=1; fi
  done
  return "$failed"
}

show_serial_status() {
  if serial_running; then
    echo "$label serial pair is running (PID $(<"$pid_file"))."
    ls -l -- "$host_device" "$mujoco_device"
  else
    echo "$label serial pair is not running."
    return 1
  fi
}

case $action in
  start)
    start_vcan
    with_serial_pair imu start_serial
    with_serial_pair power start_serial
    ;;
  stop)
    with_serial_pair imu stop_serial
    with_serial_pair power stop_serial
    echo "VCAN interfaces were left up."
    ;;
  status)
    show_status
    ;;
esac
