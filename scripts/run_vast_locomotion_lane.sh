#!/usr/bin/env bash
set -u

if (( $# != 5 )); then
  echo "Usage: $0 LANE QUEUE STATUS_LOG RESULTS_FILE BATCH_NAME" >&2
  exit 2
fi

lane=$1
queue=$2
status_log=$3
results_file=$4
batch_name=$5
max_attempts=${MAX_ATTEMPTS:-3}
retry_delay=${RETRY_DELAY:-15}

if (( max_attempts < 1 || retry_delay < 0 )); then
  echo "MAX_ATTEMPTS must be positive and RETRY_DELAY must be nonnegative" >&2
  exit 2
fi

while IFS=$'\t' read -r condition seed <&3; do
  name="${batch_name}-${condition}-seed${seed}"
  rc=1
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    printf '%s lane%s START attempt=%s %s\n' \
      "$(date -Is)" "$lane" "$attempt" "$name" | tee -a "$status_log"
    BRANCH=${BRANCH:?} \
      MAX_PRICE=${MAX_PRICE:-1.25} \
      TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-75497472} \
      RUN_PREFIX=$batch_name \
      BELLO_VAST_TRAIN_BIN=${BELLO_VAST_TRAIN_BIN:-} \
      JAX_DEFAULT_MATMUL_PRECISION=highest \
      scripts/run_locomotion_ablation.sh \
      "${RUNNER_MODE:-vast}" "$condition" "$seed" </dev/null
    rc=$?
    printf '%s lane%s END attempt=%s rc=%s %s\n' \
      "$(date -Is)" "$lane" "$attempt" "$rc" "$name" | tee -a "$status_log"
    if (( rc == 0 )); then
      break
    fi
    if (( attempt < max_attempts )); then
      sleep "$retry_delay"
    fi
  done
  printf '%s\t%s\t%s\t%s\n' "$condition" "$seed" "$rc" "$name" >> "$results_file"
done 3< "$queue"

printf '%s lane%s QUEUE COMPLETE\n' "$(date -Is)" "$lane" | tee -a "$status_log"
