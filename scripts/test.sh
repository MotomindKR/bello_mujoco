#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

test_files=("$@")
if (( ${#test_files[@]} == 0 )); then
  test_files=(tests/test_*.py)
fi

for test_file in "${test_files[@]}"; do
  echo "pytest $test_file"
  uv run pytest "$test_file"
done
