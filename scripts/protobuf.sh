#!/usr/bin/env bash
set -euo pipefail

readonly repository_root="$(cd "$(dirname "$0")/.." && pwd)"
readonly source_root="$repository_root/src"
readonly canonical_proto_path="$repository_root/deps/bello-stack/components/bello-pose-api/proto/bello/pose/v1/pose.proto"
readonly logical_proto_path="bello_mujoco/hardware/_pose.proto"
readonly generated_path="$source_root/bello_mujoco/hardware/_pose_pb2.py"
readonly action="${1:---check}"

generate_into() {
  local output_root=$1
  local proto_root=$2
  local staged_proto_path="$proto_root/$logical_proto_path"

  mkdir -p "$proto_root/bello_mujoco/hardware"
  mkdir -p "$output_root"
  cp "$canonical_proto_path" "$staged_proto_path"
  python -m grpc_tools.protoc \
    --proto_path="$proto_root" \
    --python_out="$output_root" \
    "$staged_proto_path"
}

case "$action" in
  generate|--check)
    ;;
  *)
    echo "Usage: $0 [--check|generate]" >&2
    exit 2
    ;;
esac

if [[ ! -f "$canonical_proto_path" ]]; then
  echo "Canonical pose protobuf is missing; initialize deps/bello-stack" >&2
  exit 1
fi

temporary_root=$(mktemp -d)
trap 'rm -rf "$temporary_root"' EXIT
generate_into "$temporary_root/output" "$temporary_root/proto"
candidate="$temporary_root/output/$logical_proto_path"
candidate="${candidate%.proto}_pb2.py"

case "$action" in
  generate)
    cp "$candidate" "$generated_path"
    ;;
  --check)
    if ! cmp -s "$generated_path" "$candidate"; then
      diff -u "$generated_path" "$candidate" || true
      echo "Generated protobuf is stale; run: uv run bash scripts/protobuf.sh generate" >&2
      exit 1
    fi
    ;;
esac
