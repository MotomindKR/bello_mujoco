#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 BRANCH" >&2
  exit 2
fi

branch=$1
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
git fetch origin "$branch"
commit=$(git rev-parse "origin/$branch")
timestamp=$(date -u +%Y%m%d-%H%M%S)
state_root="${XDG_STATE_HOME:-$HOME/.local/state}/bello-mujoco/differential-ablations/$timestamp"
mkdir -p "$state_root"

configs=(
  experiments/train/locomotion_berkeley_direct_short.yaml
  experiments/train/locomotion_berkeley_differential_equivalent_short.yaml
  experiments/train/locomotion_berkeley_differential_hardware_short.yaml
)
names=(direct joint-effort18 joint-effort36)

if [[ -n ${TMUX:-} ]]; then
  session=$(tmux display-message -p '#S')
  window=$(tmux new-window -d -P -F '#{window_id}' -t "$session:" -n "diff-$timestamp")
else
  session="bello-diff-$timestamp"
  tmux new-session -d -s "$session" -n "diff-$timestamp"
  window=$(tmux display-message -p -t "$session:diff-$timestamp" '#{window_id}')
fi

panes=("$(tmux display-message -p -t "$window" '#{pane_id}')")
for _ in 1 2; do
  panes+=("$(tmux split-window -d -P -F '#{pane_id}' -t "$window")")
done
tmux select-layout -t "$window" tiled
tmux set-option -w -t "$window" remain-on-exit on
tmux set-option -w -t "$window" pane-border-status top
tmux set-option -w -t "$window" pane-border-format ' #{pane_title} '

for index in "${!configs[@]}"; do
  worktree="$state_root/${names[$index]}"
  git worktree add --detach "$worktree" "$commit"
  command=$(printf \
    'cd %q; tmux select-pane -T %q; %q --branch %q --train-config %q --max-price %q --yes; exec bash' \
    "$worktree" "${names[$index]}" "$repo_root/.venv/bin/bello-vast-train" \
    "$branch" "${configs[$index]}" "${MAX_PRICE:-1.25}")
  tmux send-keys -t "${panes[$index]}" -l "$command"
  tmux send-keys -t "${panes[$index]}" Enter
done

tmux select-layout -t "$window" tiled
echo "Launched differential ablations from $commit in $session:diff-$timestamp"
echo "State/worktrees: $state_root"
