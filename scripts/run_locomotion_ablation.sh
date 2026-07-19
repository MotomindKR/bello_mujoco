#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_locomotion_ablation.sh --list
  scripts/run_locomotion_ablation.sh print CONDITION SEED [HYDRA_OVERRIDE ...]
  scripts/run_locomotion_ablation.sh local CONDITION SEED [HYDRA_OVERRIDE ...]
  scripts/run_locomotion_ablation.sh vast  CONDITION SEED [HYDRA_OVERRIDE ...]

Environment:
  TOTAL_TIMESTEPS       Defaults to 75497472 (1,536 updates).
  RUN_PREFIX            Defaults to locomotion-gait-ablation.
  BRANCH                 Vast branch; defaults to the current branch.
  MAX_PRICE              Vast maximum hourly price; defaults to 1.25.
  WANDB_MODE             Local W&B mode; defaults to online.
  BELLO_VAST_TRAIN_BIN   Optional absolute bello-vast-train executable.
EOF
}

list_conditions() {
  cat <<'EOF'
base                 retained stance-only control
debounce_03          30 ms contact on/off debounce
debounce_04          40 ms contact on/off debounce
debounce_05          50 ms contact on/off debounce
timing_05            40 ms debounce + premature-transition weight -0.05
timing_10            40 ms debounce + premature-transition weight -0.10
timing_20            40 ms debounce + premature-transition weight -0.20
slip_005             stance-foot slip weight -0.05
slip_010             stance-foot slip weight -0.10
clock                 1.0 s clock + gait-contact 0.35 + 40 ms debounce
clock_swing_025       clock + scheduled swing velocity 0.25
clock_swing_050       clock + scheduled swing velocity 0.50
clock_timing          clock + premature-transition weight -0.10
full_low              full candidate: timing -0.10, swing 0.25, slip -0.05
full_high             full candidate: timing -0.10, swing 0.50, slip -0.10
berkeley_hard         hard joint-limit backstop only
berkeley_joint90      90% physical joint limits, no soft action penalty
berkeley_action08     hard joint limits + soft action penalty above 80%
berkeley_joint90_action08  90% joint limits + soft action penalty above 80%
EOF
}

if [[ ${1:-} == --list ]]; then
  list_conditions
  exit 0
fi
if (( $# < 3 )); then
  usage >&2
  exit 2
fi

mode=$1
condition=$2
seed=$3
shift 3
extra_overrides=("$@")

repo_root=$(git rev-parse --show-toplevel)
config_name=locomotion_gait_base
overrides=()

debounce() {
  overrides+=(
    "env_config.contact.foot_contact_on_seconds=$1"
    "env_config.contact.foot_contact_off_seconds=$1"
  )
}

timing() {
  debounce 0.04
  overrides+=(
    "reward_config.contact_timing.enabled=true"
    "reward_config.contact_timing.weight=-$1"
  )
}

clock_only() {
  config_name=locomotion_gait
  overrides+=(
    "reward_config.contact_timing.enabled=false"
    "reward_config.swing_foot_velocity.enabled=false"
    "reward_config.foot_slip.enabled=false"
  )
}

berkeley() {
  config_name=locomotion_berkeley
  overrides+=(
    "reward_config.joint_limit.soft_limit_factor=1.0"
    "reward_config.action_limit.enabled=false"
  )
}

case "$condition" in
  base) ;;
  debounce_03) debounce 0.03 ;;
  debounce_04) debounce 0.04 ;;
  debounce_05) debounce 0.05 ;;
  timing_05) timing 0.05 ;;
  timing_10) timing 0.10 ;;
  timing_20) timing 0.20 ;;
  slip_005)
    overrides+=("reward_config.foot_slip.enabled=true" "reward_config.foot_slip.weight=-0.05")
    ;;
  slip_010)
    overrides+=("reward_config.foot_slip.enabled=true" "reward_config.foot_slip.weight=-0.10")
    ;;
  clock) clock_only ;;
  clock_swing_025)
    clock_only
    overrides+=("reward_config.swing_foot_velocity.enabled=true" "reward_config.swing_foot_velocity.weight=0.25")
    ;;
  clock_swing_050)
    clock_only
    overrides+=("reward_config.swing_foot_velocity.enabled=true" "reward_config.swing_foot_velocity.weight=0.50")
    ;;
  clock_timing)
    clock_only
    overrides+=("reward_config.contact_timing.enabled=true" "reward_config.contact_timing.weight=-0.10")
    ;;
  full_low) config_name=locomotion_gait ;;
  full_high)
    config_name=locomotion_gait
    overrides+=("reward_config.swing_foot_velocity.weight=0.50" "reward_config.foot_slip.weight=-0.10")
    ;;
  berkeley_hard) berkeley ;;
  berkeley_joint90)
    berkeley
    overrides+=("reward_config.joint_limit.soft_limit_factor=0.9")
    ;;
  berkeley_action08)
    berkeley
    overrides+=("reward_config.action_limit.enabled=true")
    ;;
  berkeley_joint90_action08)
    berkeley
    overrides+=(
      "reward_config.joint_limit.soft_limit_factor=0.9"
      "reward_config.action_limit.enabled=true"
    )
    ;;
  *)
    echo "Unknown condition: $condition" >&2
    list_conditions >&2
    exit 2
    ;;
esac

total_timesteps=${TOTAL_TIMESTEPS:-75497472}
run_prefix=${RUN_PREFIX:-locomotion-gait-ablation}
run_name="${run_prefix}-${condition}-seed${seed}"
common_overrides=(
  "config.seed=$seed"
  "config.total_timesteps=$total_timesteps"
  "checkpoint_config.name=$run_name"
  "wandb_config.run_name=$run_name"
  "${overrides[@]}"
  "${extra_overrides[@]}"
)

case "$mode" in
  print)
    printf 'config=%s\nrun_name=%s\n' "$config_name" "$run_name"
    printf 'override=%s\n' "${common_overrides[@]}"
    ;;
  local)
    exec uv run bello-train-ppo \
      --config-dir "$repo_root/experiments/train" \
      --config-name "$config_name" \
      dry_run=false \
      wandb_config.enabled=true \
      "wandb_config.mode=${WANDB_MODE:-online}" \
      checkpoint_config.enabled=true \
      "${common_overrides[@]}"
    ;;
  vast)
    branch=${BRANCH:-$(git branch --show-current)}
    if [[ -z $branch ]]; then
      echo "BRANCH is required from a detached worktree" >&2
      exit 2
    fi
    train_config="experiments/train/${config_name}.yaml"
    if [[ -n ${BELLO_VAST_TRAIN_BIN:-} ]]; then
      vast_command=("$BELLO_VAST_TRAIN_BIN")
    else
      vast_command=(uv run bello-vast-train)
    fi
    exec "${vast_command[@]}" --yes \
      --branch "$branch" \
      --train-config "$train_config" \
      --max-price "${MAX_PRICE:-1.25}" \
      --run-name "$run_name" \
      -- "${common_overrides[@]}"
    ;;
  *)
    echo "Mode must be print, local, or vast, got: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
