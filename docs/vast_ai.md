# Vast.ai Workflow

This workflow launches an RTX 5090 Vast.ai instance, bootstraps the repo with
`uv`, and runs PPO training with W&B metrics and video logging. It intentionally
does not use Nix on the remote host and does not upload checkpoints as W&B
artifacts.

## Prerequisites

Install the Vast CLI locally or let the scripts run it through `uvx`:

```bash
uv tool install vastai
vastai set api-key <VAST_API_KEY>
vastai create ssh-key ~/.ssh/id_ed25519.pub
```

Training uses W&B:

```bash
export WANDB_API_KEY=<WANDB_API_KEY>
export WANDB_PROJECT=bello-mujoco
```

## Launch

Dry-run the launch command:

```bash
scripts/vast/launch.sh --dry-run
```

Launch a default RTX 5090 instance:

```bash
scripts/vast/launch.sh --repo-ref main
```

Defaults:

```text
gpu_name=RTX_5090
num_gpus=1
verified=true
direct_port_count>=1
rentable=true
cpu_ram>=64
cuda_vers>=13.0
disk=80GB
image=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
```

The created instance id is stored in `.vast/instance_id`. The remote repo path is
`/workspace/bello_mujoco`.

## Remote Setup

The Vast on-start command installs minimal OS packages, clones the repo, checks
out the requested ref, and runs:

```bash
scripts/vast/remote_setup.sh
```

Remote setup exports the useful environment from the local flake without using
Nix:

```bash
UV_CACHE_DIR="$PWD/.uv-cache"
UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
WARP_CACHE_PATH="$PWD/.warp-cache"
XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
```

Then it runs:

```bash
uv python install 3.12
uv sync
uv run bello-train-ppo dry_run=true config.num_envs=1 config.num_steps=1 config.num_minibatches=1
```

## Train

Start training in a remote `tmux` session:

```bash
WANDB_API_KEY=$WANDB_API_KEY scripts/vast/train.sh
```

Pass Hydra overrides after `--`:

```bash
WANDB_API_KEY=$WANDB_API_KEY scripts/vast/train.sh -- \
  config.total_timesteps=1048576 \
  video_config.enabled=false
```

The default config is `configs/train/vast_rtx5090.yaml`. It enables W&B,
checkpointing, and videos. Checkpoints are written locally on the instance under
`checkpoints/`; upload or backup of those files is intentionally left to a
separate workflow.

The training command exports:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Monitor

Show Vast state, SSH URL, and the remote tmux/log tail:

```bash
scripts/vast/status.sh
```

W&B metrics and videos appear in the configured W&B project. The remote training
log is also written under:

```text
/workspace/bello_mujoco/outputs/vast/
```

## Destroy

Destroying removes the remote instance and all files on it:

```bash
scripts/vast/destroy.sh
```

Use noninteractive teardown in scripts:

```bash
scripts/vast/destroy.sh --yes
```

## OOM Guidance

The default Vast preset uses `4096 * 256 = 1048576` samples per PPO update. With
large observations or observation history, the rollout buffer can consume several
GiB before JAX/XLA, MuJoCo-Warp, W&B, and checkpoint overhead. The launcher
therefore requires `cpu_ram>=64` by default.

If a host still runs out of CPU RAM, reduce one of:

```bash
config.num_envs=2048
config.num_steps=128
video_config.enabled=false
```

Do not run large validation jobs on the same machine while training.
