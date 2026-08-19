#!/usr/bin/env bash
set -euo pipefail

readonly repository_root="$(cd "$(dirname "$0")/.." && pwd)"
readonly source_root="$repository_root/src"
readonly canonical_pose_proto="$repository_root/deps/bello-stack/components/bello-pose-api/proto/bello/pose/v1/pose.proto"
readonly canonical_remote_proto="$repository_root/deps/bello-stack/components/bello-remote-api/proto/bello/remote/v1/remote.proto"
readonly logical_pose_proto="bello_mujoco/hardware/_pose.proto"
readonly logical_remote_proto="bello_mujoco/hardware/_remote.proto"
readonly generated_pose_path="$source_root/bello_mujoco/hardware/_pose_pb2.py"
readonly generated_remote_path="$source_root/bello_mujoco/hardware/_remote_pb2.py"
readonly action="${1:---check}"

generate_into() {
  local canonical_proto=$1
  local logical_proto=$2
  local output_root=$3
  local proto_root=$4
  local staged_proto_path="$proto_root/$logical_proto"

  mkdir -p "$(dirname "$staged_proto_path")"
  mkdir -p "$output_root"
  cp "$canonical_proto" "$staged_proto_path"
  python -m grpc_tools.protoc \
    --proto_path="$proto_root" \
    --python_out="$output_root" \
    "$staged_proto_path"
}

check_binding() {
  local generated=$1
  local candidate=$2

  if ! cmp -s "$generated" "$candidate"; then
    diff -u "$generated" "$candidate" || true
    echo "Generated protobuf is stale; run: uv run bash scripts/protobuf.sh generate" >&2
    exit 1
  fi
}

candidate_for() {
  local output_root=$1
  local logical_proto=$2

  local candidate="$output_root/$logical_proto"
  echo "${candidate%.proto}_pb2.py"
}

case "$action" in
  generate|--check)
    ;;
  *)
    echo "Usage: $0 [--check|generate]" >&2
    exit 2
    ;;
esac

for canonical_proto in "$canonical_pose_proto" "$canonical_remote_proto"; do
  if [[ ! -f "$canonical_proto" ]]; then
    echo "Canonical protobuf is missing: $canonical_proto" >&2
    echo "Initialize deps/bello-stack at the pinned revision" >&2
    exit 1
  fi
done

temporary_root=$(mktemp -d)
trap 'rm -rf "$temporary_root"' EXIT
generate_into "$canonical_pose_proto" "$logical_pose_proto" \
  "$temporary_root/output" "$temporary_root/proto"
generate_into "$canonical_remote_proto" "$logical_remote_proto" \
  "$temporary_root/output" "$temporary_root/proto"
candidate_pose=$(candidate_for "$temporary_root/output" "$logical_pose_proto")
candidate_remote=$(candidate_for "$temporary_root/output" "$logical_remote_proto")

case "$action" in
  generate)
    cp "$candidate_pose" "$generated_pose_path"
    cp "$candidate_remote" "$generated_remote_path"
    ;;
  --check)
    check_binding "$generated_pose_path" "$candidate_pose"
    check_binding "$generated_remote_path" "$candidate_remote"
    ;;
esac
