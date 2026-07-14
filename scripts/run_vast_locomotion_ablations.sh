#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Launch the locomotion matrix in parallel tmux lanes.

Usage:
  scripts/run_vast_locomotion_ablations.sh BRANCH

Environment:
  MAX_PARALLEL     Concurrent Vast instances, default 6.
  MAX_PRICE        Per-instance hourly ceiling, default 1.25.
  SEEDS            Space-separated seeds, default "0 1 2".
  CONDITIONS       Space-separated condition names; defaults to the full matrix.
  TOTAL_TIMESTEPS  Defaults to 75497472 (1,536 updates).
  RUN_PREFIX       Optional W&B/checkpoint prefix; a timestamp is appended.
  DRY_RUN          Set to 1 to exercise worktrees, queues, and tmux without rent.

The launcher creates one detached Git worktree per lane so each blocking Vast
controller has independent .vast lock and teardown state. Worktrees and logs
live under ${XDG_STATE_HOME:-$HOME/.local/state}/bello-mujoco/vast-ablations.
EOF
}

if (( $# != 1 )); then
  usage >&2
  exit 2
fi

branch=$1
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
git fetch origin "$branch"
commit=$(git rev-parse "origin/$branch")
max_parallel=${MAX_PARALLEL:-6}
if (( max_parallel < 1 )); then
  echo "MAX_PARALLEL must be positive" >&2
  exit 2
fi

read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a conditions <<< "${CONDITIONS:-base debounce_03 debounce_04 debounce_05 timing_05 timing_10 timing_20 slip_005 slip_010 clock clock_swing_025 clock_swing_050 clock_timing full_low full_high}"
if (( ${#seeds[@]} == 0 || ${#conditions[@]} == 0 )); then
  echo "SEEDS and CONDITIONS must not be empty" >&2
  exit 2
fi

timestamp=$(date -u +%Y%m%d-%H%M%S)
batch_name="${RUN_PREFIX:-locomotion-gait}-${timestamp}"
window_name="gait-$timestamp"
state_root="${XDG_STATE_HOME:-$HOME/.local/state}/bello-mujoco/vast-ablations/$batch_name"
mkdir -p "$state_root"
status_log="$state_root/status.log"
: > "$status_log"
runner_mode=vast
if [[ ${DRY_RUN:-0} == 1 ]]; then
  runner_mode=print
fi

job_count=$((${#seeds[@]} * ${#conditions[@]}))
lane_count=$max_parallel
if (( lane_count > job_count )); then
  lane_count=$job_count
fi

for ((lane = 1; lane <= lane_count; lane++)); do
  worktree="$state_root/lane$lane"
  git worktree add --detach "$worktree" "$commit"
  : > "$state_root/lane$lane.queue"
done

job_index=0
for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    lane=$((job_index % lane_count + 1))
    printf '%s\t%s\n' "$condition" "$seed" >> "$state_root/lane$lane.queue"
    ((job_index += 1))
  done
done

if [[ -n ${TMUX:-} ]]; then
  session=$(tmux display-message -p '#S')
  window=$(tmux new-window -d -P -F '#{window_id}' -t "$session:" -n "$window_name")
else
  session="bello-gait-$timestamp"
  tmux new-session -d -s "$session" -n "$window_name"
  window=$(tmux display-message -p -t "$session:$window_name" '#{window_id}')
fi

first_pane=$(tmux display-message -p -t "$window" '#{pane_id}')
panes=("$first_pane")
for ((lane = 2; lane <= lane_count; lane++)); do
  panes+=("$(tmux split-window -d -P -F '#{pane_id}' -t "$window" -c "$state_root/lane$lane")")
done
tmux select-layout -t "$window" tiled
tmux set-option -w -t "$window" remain-on-exit on
tmux set-option -w -t "$window" pane-border-status top
tmux set-option -w -t "$window" pane-border-format ' #{pane_index}: #{pane_title} '

for ((lane = 1; lane <= lane_count; lane++)); do
  worktree="$state_root/lane$lane"
  queue="$state_root/lane$lane.queue"
  pane=${panes[$((lane - 1))]}
  command=$(printf \
    'set +e; cd %q; tmux select-pane -T %q; while IFS=$'"'"'\\t'"'"' read -r condition seed; do name=%q-"$condition"-seed"$seed"; printf "%%s lane%d START %%s\\n" "$(date -Is)" "$name" | tee -a %q; BRANCH=%q MAX_PRICE=%q TOTAL_TIMESTEPS=%q RUN_PREFIX=%q BELLO_VAST_TRAIN_BIN=%q scripts/run_locomotion_ablation.sh %q "$condition" "$seed"; rc=$?; printf "%%s lane%d END rc=%%s %%s\\n" "$(date -Is)" "$rc" "$name" | tee -a %q; done < %q; printf "%%s lane%d QUEUE COMPLETE\\n" "$(date -Is)" | tee -a %q; exec bash' \
    "$worktree" "lane$lane" "$batch_name" "$lane" "$status_log" \
    "$branch" "${MAX_PRICE:-1.25}" "${TOTAL_TIMESTEPS:-75497472}" "$batch_name" \
    "$repo_root/.venv/bin/bello-vast-train" "$runner_mode" "$lane" "$status_log" \
    "$queue" "$lane" "$status_log")
  tmux send-keys -t "$pane" -l "$command"
  tmux send-keys -t "$pane" Enter
done
tmux select-layout -t "$window" tiled

cat <<EOF
Launched $job_count jobs across $lane_count parallel $runner_mode lanes.
Commit: $commit
Window: $session:$window_name
Status: $status_log

Open the grid:
  tmux select-window -t $session:$window_name

Follow all queue transitions:
  tail -f $status_log
EOF
