# Setup

## Nix Shell

Enter the dev shell:

```bash
nix develop
```

The flake installs only:

```text
git
uv
```

The shell hook sets repo-local cache paths:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export WARP_CACHE_PATH="$PWD/.warp-cache"
```

Then it runs:

```bash
uv python install 3.12
uv sync
source .venv/bin/activate
```

## Direnv

`.envrc` contains:

```bash
use flake
```

Enable it once per checkout:

```bash
direnv allow
```

After that, entering the repo should expose `uv`, activate `.venv`, and provide `python`.

## CUDA

JAX CUDA libraries come from the `jax[cuda13]` Python wheel. The flake does not inject CUDA or XLA library paths. The expected runtime path is:

```text
Python wheel CUDA runtime + system driver
```

Smoke check:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env --step
```

Expected output includes:

```text
backend: gpu
mjx_impl: warp
```

## Local Artifacts

Do not commit these:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
.wandb/
checkpoints/
outputs/
/tmp/bello-mujoco-assets/
/tmp/bello-mujoco-test-assets-*
```

Generated assets are reproducible from tracked source assets and config.
