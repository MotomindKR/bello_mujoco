#!/usr/bin/env bash
set -euo pipefail

readonly repository_root="$(cd "$(dirname "$0")/.." && pwd)"
readonly source_root="$repository_root/src"
readonly proto_path="$source_root/bello_mujoco/hardware/_pose.proto"
readonly generated_path="$source_root/bello_mujoco/hardware/_pose_pb2.py"
readonly action="${1:---check}"

generate_into() {
  local output_root=$1
  python -m grpc_tools.protoc \
    --proto_path="$source_root" \
    --python_out="$output_root" \
    "$proto_path"
}

case "$action" in
  generate)
    generate_into "$source_root"
    ;;
  --check)
    temporary_root=$(mktemp -d)
    trap 'rm -rf "$temporary_root"' EXIT
    generate_into "$temporary_root"
    candidate="$temporary_root/bello_mujoco/hardware/_pose_pb2.py"
    if ! cmp -s "$generated_path" "$candidate"; then
      diff -u "$generated_path" "$candidate" || true
      echo "Generated protobuf is stale; run: uv run bash scripts/protobuf.sh generate" >&2
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [--check|generate]" >&2
    exit 2
    ;;
esac
