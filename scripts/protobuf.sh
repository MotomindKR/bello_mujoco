#!/usr/bin/env bash
set -euo pipefail

readonly repository_root="$(cd "$(dirname "$0")/.." && pwd)"
readonly source_root="$repository_root/src"
readonly canonical_pose_proto="$repository_root/deps/bello-stack/components/bello-pose-api/proto/bello/pose/v1/pose.proto"
readonly canonical_operator_proto="$repository_root/rust/bello-policy/proto/operator.proto"
readonly canonical_smplx_proto="$repository_root/deps/pico-smplx-bridge/pose_stream.proto"
readonly canonical_reference_proto="$repository_root/rust/bello-policy/proto/reference.proto"
readonly logical_pose_proto="bello_mujoco/hardware/_pose.proto"
readonly logical_operator_proto="bello_mujoco/hardware/_operator.proto"
readonly logical_smplx_proto="bello_mujoco/hardware/_smplx.proto"
readonly generated_pose_path="$source_root/bello_mujoco/hardware/_pose_pb2.py"
readonly generated_operator_path="$source_root/bello_mujoco/hardware/_operator_pb2.py"
readonly generated_smplx_path="$source_root/bello_mujoco/hardware/_smplx_pb2.py"
readonly generated_reference_path="$source_root/bello_mujoco/hardware/reference_pb2.py"
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

for canonical_proto in "$canonical_pose_proto" "$canonical_operator_proto" "$canonical_smplx_proto" "$canonical_reference_proto"; do
  if [[ ! -f "$canonical_proto" ]]; then
    echo "Canonical protobuf is missing: $canonical_proto" >&2
    echo "Initialize the pinned deps/bello-stack and deps/pico-smplx-bridge submodules" >&2
    exit 1
  fi
done

temporary_root=$(mktemp -d)
trap 'rm -rf "$temporary_root"' EXIT
generate_into "$canonical_pose_proto" "$logical_pose_proto" \
  "$temporary_root/output" "$temporary_root/proto"
generate_into "$canonical_operator_proto" "$logical_operator_proto" \
  "$temporary_root/output" "$temporary_root/proto"
generate_into "$canonical_smplx_proto" "$logical_smplx_proto" \
  "$temporary_root/output" "$temporary_root/proto"
generate_into "$canonical_reference_proto" "reference.proto" \
  "$temporary_root/output" "$temporary_root/proto"
candidate_pose=$(candidate_for "$temporary_root/output" "$logical_pose_proto")
candidate_operator=$(candidate_for "$temporary_root/output" "$logical_operator_proto")
candidate_smplx=$(candidate_for "$temporary_root/output" "$logical_smplx_proto")

case "$action" in
  generate)
    cp "$candidate_pose" "$generated_pose_path"
    cp "$candidate_operator" "$generated_operator_path"
    cp "$candidate_smplx" "$generated_smplx_path"
    cp "$temporary_root/output/reference_pb2.py" "$generated_reference_path"
    ;;
  --check)
    check_binding "$generated_pose_path" "$candidate_pose"
    check_binding "$generated_operator_path" "$candidate_operator"
    check_binding "$generated_smplx_path" "$candidate_smplx"
    check_binding "$generated_reference_path" "$temporary_root/output/reference_pb2.py"
    ;;
esac
