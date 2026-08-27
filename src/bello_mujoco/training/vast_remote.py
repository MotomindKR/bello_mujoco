"""Direct-SSH command construction and detached Vast training execution."""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bello_mujoco.training.vast_provider import VastApi

if TYPE_CHECKING:
    from bello_mujoco.training.vast import GitSelection, VastTrainConfig


SSH_SUBPROCESS_TIMEOUT_SECONDS = 60.0
NONRECOVERABLE_INSTANCE_STATUSES = frozenset(("exited", "offline"))
EPHEMERAL_KNOWN_HOSTS_PATH = (
    Path(tempfile.gettempdir()) / f"bello-mujoco-vast-{os.getpid()}.known_hosts"
)


def _remove_ephemeral_known_hosts() -> None:
    EPHEMERAL_KNOWN_HOSTS_PATH.unlink(missing_ok=True)


atexit.register(_remove_ephemeral_known_hosts)


def _require_recoverable_instance(
    api: VastApi | None,
    instance_id: int | None,
) -> None:
    if api is None or instance_id is None:
        return
    try:
        info = api.show_instance(instance_id)
    except Exception:
        print(
            "Vast instance status is temporarily unavailable; continuing SSH recovery."
        )
        return
    if info is None:
        status = "absent"
    else:
        status = str(info.get("actual_status", "")).strip()
        if status not in NONRECOVERABLE_INSTANCE_STATUSES:
            return
    raise RuntimeError(
        f"Vast instance {instance_id} entered non-recoverable state "
        f"{status!r} while SSH was unavailable"
    )


@dataclass(frozen=True)
class RemoteJobUpdate:
    """One bounded log/status response from a detached remote job."""

    state: str
    offset: int
    output: bytes
    returncode: int | None = None


def direct_ssh_target(
    info: Mapping[str, Any] | None,
) -> tuple[str, str, int] | None:
    if info is None:
        return None
    host = str(info.get("public_ipaddr", "")).strip()
    ports = info.get("ports")
    if not host or not isinstance(ports, Mapping):
        return None
    mappings = ports.get("22/tcp")
    if not isinstance(mappings, list):
        return None
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        try:
            port = int(mapping.get("HostPort"))
        except (TypeError, ValueError):
            continue
        if 0 < port <= 65535:
            return "root", host, port
    return None


def ssh_command(target: tuple[str, str, int], remote_command: str) -> list[str]:
    user, host, port = target
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"UserKnownHostsFile={EPHEMERAL_KNOWN_HOSTS_PATH}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(port),
        f"{user}@{host}",
        remote_command,
    ]


class SshExecutor:
    """Subprocess boundary for non-interactive SSH probes and commands."""

    def probe(self, target: tuple[str, str, int], *, timeout_seconds: float) -> None:
        subprocess.run(
            ssh_command(target, "true"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )

    def run(
        self,
        target: tuple[str, str, int],
        remote_command: str,
        *,
        input_text: str | None = None,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        kwargs: dict[str, Any] = {
            "text": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if input_text is not None:
            kwargs["input"] = input_text
        return subprocess.run(
            ssh_command(target, remote_command),
            timeout=timeout_seconds,
            **kwargs,
        )


class RsyncExecutor:
    """Resumable local-to-instance transfer boundary."""

    def run(self, command: list[str], *, timeout_seconds: float) -> None:
        subprocess.run(command, check=True, timeout=timeout_seconds)


def wait_for_ssh(
    api: VastApi,
    instance_id: int,
    timeout_seconds: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
    executor: SshExecutor | None = None,
) -> tuple[str, str, int]:
    if executor is None:
        executor = SshExecutor()
    # Vast frequently reuses public IP/port pairs for unrelated machines. Keep
    # TOFU host-key verification within one rental instead of conflicting with
    # the user's durable SSH trust store or a previous offer in this process.
    _remove_ephemeral_known_hosts()
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            target = direct_ssh_target(api.show_instance(instance_id))
        except OSError as exc:
            last_error = exc
        else:
            if target is None:
                last_error = RuntimeError("direct SSH mapping is not available")
            else:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    executor.probe(
                        target,
                        timeout_seconds=min(
                            SSH_SUBPROCESS_TIMEOUT_SECONDS,
                            remaining,
                        ),
                    )
                    return target
                except (
                    OSError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    last_error = exc
        if time.monotonic() < deadline:
            sleep(10)
    raise TimeoutError(f"direct SSH did not become ready: {last_error}")


def measure_upload_mib_per_second(
    target: tuple[str, str, int],
    *,
    size_mib: int,
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> float:
    """Measure the direct-SSH upload route before staging large inputs."""

    payload = bytes(size_mib * 1024 * 1024)
    started_at = time.monotonic()
    try:
        result = runner(
            ssh_command(target, "cat >/dev/null"),
            input=payload,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=min(timeout_seconds, 60),
        )
    except subprocess.TimeoutExpired:
        return 0.0
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        raise RuntimeError(f"Vast upload route probe failed: {stderr.strip()}")
    elapsed = max(time.monotonic() - started_at, 1e-6)
    return size_mib / elapsed


REMOTE_TRAIN_SCRIPT = r"""set -Eeuo pipefail

branch="$1"
expected_commit="$2"
repo_url="$3"
remote_dir="$4"
experiment="$5"
motion_path="$6"
end_effector_payload_mass_kg="$7"
run_name="$8"
dataset_path="$9"
teacher_checkpoint_path="${10}"
warm_start_checkpoint_path="${11}"
allow_git_commit_mismatch="${12}"
sonic_artifact_path="${13}"
resume_checkpoint_path="${14}"
total_timesteps="${15}"
num_envs="${16}"
checkpoint_every_updates="${17}"
learning_rate="${18}"
result_dir="${19}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates ffmpeg git libegl1 libgl1 libglib2.0-0 libxrender1 python3 python3-pip rsync

python3 -m pip install --break-system-packages --disable-pip-version-check uv==0.11.25
command -v uv >/dev/null 2>&1 || { echo "uv installation failed" >&2; exit 1; }

askpass="$(mktemp /tmp/bello-git-askpass.XXXXXX)"
cleanup_askpass() { rm -f "$askpass"; }
trap cleanup_askpass EXIT
cat > "$askpass" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *) printf '%s\n' "${GITHUB_TOKEN:-}" ;;
esac
ASKPASS
chmod 700 "$askpass"

rm -rf "$remote_dir"
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git clone --recurse-submodules --single-branch --branch "$branch" "$repo_url" "$remote_dir"
submodule_status="$(git -C "$remote_dir" submodule status --recursive)"
if [ -z "$submodule_status" ] || printf '%s\n' "$submodule_status" | grep -Eq '^[-+U]'; then
  echo "a required submodule is absent, uninitialized, or not at its pinned commit" >&2
  exit 1
fi
unset GITHUB_TOKEN
cleanup_askpass
trap - EXIT
cd "$remote_dir"
actual_commit="$(git rev-parse HEAD)"
if [ "$actual_commit" != "$expected_commit" ]; then
  echo "origin/$branch moved from $expected_commit to $actual_commit; refusing unpinned training" >&2
  exit 1
fi

export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export WARP_CACHE_PATH="$PWD/.warp-cache"
export MUJOCO_GL=egl
export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

uv python install
uv sync --frozen
uv run python -c 'import jax; devices=jax.devices(); print({"backend": jax.default_backend(), "devices": [str(x) for x in devices]}); assert jax.default_backend() == "gpu"'

training_overrides=("experiment=$experiment")
if [ -n "$motion_path" ]; then
  training_overrides+=("motion_path=$motion_path")
fi
if [ -n "$end_effector_payload_mass_kg" ]; then
  training_overrides+=("end_effector_payload_mass_kg=$end_effector_payload_mass_kg")
fi
if [ -n "$dataset_path" ]; then
  training_overrides+=("dataset_path=$dataset_path")
fi
if [ -n "$teacher_checkpoint_path" ]; then
  training_overrides+=("teacher_checkpoint_path=$teacher_checkpoint_path")
fi
if [ -n "$warm_start_checkpoint_path" ]; then
  training_overrides+=("warm_start_path=$warm_start_checkpoint_path")
fi
if [ -n "$sonic_artifact_path" ]; then
  training_overrides+=("sonic_artifact_path=$sonic_artifact_path")
fi
if [ "$allow_git_commit_mismatch" = "true" ] && [ "$experiment" != "sonic" ]; then
  training_overrides+=("allow_git_commit_mismatch=true")
fi
if [ -n "$total_timesteps" ]; then
  training_overrides+=("total_timesteps=$total_timesteps")
fi
if [ -n "$learning_rate" ]; then
  training_overrides+=("learning_rate=$learning_rate")
fi

actual_overrides=("${training_overrides[@]}")
if [ -n "$resume_checkpoint_path" ]; then
  actual_overrides+=("resume_path=$resume_checkpoint_path")
  if [ "$allow_git_commit_mismatch" = "true" ]; then
    actual_overrides+=("allow_git_commit_mismatch=true")
  fi
fi
if [ -n "$num_envs" ]; then
  actual_overrides+=("num_envs=$num_envs")
fi
if [ -n "$checkpoint_every_updates" ]; then
  actual_overrides+=("checkpoint_every_updates=$checkpoint_every_updates")
fi

preflight_overrides=("${training_overrides[@]}" wandb_enabled=false)
if [ "$experiment" = "sonic" ]; then
  benchmark_envs="${num_envs:-4096}"
  benchmark_updates=3
  benchmark_timesteps=$((benchmark_envs * 24 * benchmark_updates))
  preflight_overrides+=(
    "total_timesteps=$benchmark_timesteps"
    "num_envs=$benchmark_envs"
    checkpoint_enabled=false
    sonic_pretraining_enabled=false
  )
else
  preflight_overrides+=(
    total_timesteps=768
    num_envs=32
    "checkpoint_name=preflight-$run_name"
  )
fi
if [[ "$experiment" != twist_* && "$experiment" != twist2 && "$experiment" != sonic ]]; then
  preflight_overrides+=(video_enabled=false)
fi

if [ "$experiment" = "sonic" ]; then
  benchmark_gpu_log="$(mktemp /tmp/bello-sonic-gpu.XXXXXX.csv)"
  benchmark_power_log="$(mktemp /tmp/bello-sonic-power.XXXXXX.csv)"
  nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    --loop-ms=250 > "$benchmark_gpu_log" 2>/dev/null &
  benchmark_monitor_pid=$!
  nvidia-smi \
    --query-gpu=power.draw \
    --format=csv,noheader,nounits \
    --loop-ms=250 > "$benchmark_power_log" 2>/dev/null &
  benchmark_power_monitor_pid=$!
  benchmark_started_at=$SECONDS
  set +e
  uv run bello-train "${preflight_overrides[@]}"
  benchmark_status=$?
  set -e
  kill "$benchmark_monitor_pid" 2>/dev/null || true
  wait "$benchmark_monitor_pid" 2>/dev/null || true
  kill "$benchmark_power_monitor_pid" 2>/dev/null || true
  wait "$benchmark_power_monitor_pid" 2>/dev/null || true
  nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    >> "$benchmark_gpu_log" 2>/dev/null || true
  nvidia-smi \
    --query-gpu=power.draw \
    --format=csv,noheader,nounits \
    >> "$benchmark_power_log" 2>/dev/null || true
  benchmark_elapsed=$((SECONDS - benchmark_started_at))
  mkdir -p "$result_dir"
  uv run python - "$benchmark_gpu_log" "$benchmark_power_log" \
    "$benchmark_envs" \
    "$benchmark_updates" "$benchmark_elapsed" "$benchmark_status" \
    "$result_dir" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

path = pathlib.Path(sys.argv[1])
rows = []
with path.open(newline="") as source:
    for row in csv.reader(source):
        if len(row) != 3:
            continue
        try:
            utilization = float(row[0].strip())
            memory_used = float(row[1].strip())
            memory_total = float(row[2].strip())
        except ValueError:
            continue
        rows.append((utilization, memory_used, memory_total))
utilization = [row[0] for row in rows]
memory_used = [row[1] for row in rows]
memory_total = [row[2] for row in rows]
power = []
with pathlib.Path(sys.argv[2]).open(newline="") as source:
    for row in csv.reader(source):
        if not row:
            continue
        try:
            power.append(float(row[0].strip()))
        except ValueError:
            continue
elapsed = max(int(sys.argv[5]), 1)
num_envs = int(sys.argv[3])
updates = int(sys.argv[4])
active_utilization = [value for value in utilization if value > 0.0]
report = {
    "sonic_benchmark": {
        "num_envs": num_envs,
        "updates": updates,
        "elapsed_seconds": elapsed,
        "exit_status": int(sys.argv[6]),
        "samples": len(rows),
        "gpu_telemetry_available": bool(rows),
        "updates_per_second_including_startup": updates / elapsed,
        "transitions_per_second_including_startup": num_envs * 24 * updates / elapsed,
        "gpu_utilization_mean_percent": (
            statistics.fmean(utilization) if utilization else None
        ),
        "gpu_utilization_active_mean_percent": (
            statistics.fmean(active_utilization) if active_utilization else None
        ),
        "gpu_active_sample_fraction": (
            len(active_utilization) / len(utilization) if utilization else None
        ),
        "gpu_utilization_peak_percent": max(utilization, default=None),
        "gpu_memory_peak_mib": max(memory_used, default=None),
        "gpu_memory_total_mib": max(memory_total, default=None),
        "gpu_power_mean_watts": statistics.fmean(power) if power else None,
    }
}
payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
(pathlib.Path(sys.argv[7]) / "benchmark.json").write_text(payload)
print(payload, end="")
PY
  rm -f "$benchmark_gpu_log" "$benchmark_power_log"
  if [ "$benchmark_status" -ne 0 ]; then
    exit "$benchmark_status"
  fi
else
  uv run bello-train "${preflight_overrides[@]}"
fi

uv run bello-train \
  "${actual_overrides[@]}" \
  wandb_enabled=true \
  run_name="$run_name" \
  checkpoint_name="$run_name"

if [ "$experiment" = "sonic" ]; then
  mkdir -p "$result_dir"
  shopt -s nullglob
  checkpoint_roots=(checkpoints/"${run_name}"_*)
  shopt -u nullglob
  if [ "${#checkpoint_roots[@]}" -ne 1 ] || [ ! -d "${checkpoint_roots[0]}/final" ]; then
    echo "SONIC final checkpoint could not be resolved uniquely" >&2
    exit 1
  fi
  final_checkpoint="${checkpoint_roots[0]}/final"
  printf '%s\n' "$final_checkpoint" > "$result_dir/checkpoint_path.txt"
  uv run bello-eval \
    task=sonic \
    checkpoint_path="$final_checkpoint" \
    sonic_artifact_path="$sonic_artifact_path" \
    output_path="$result_dir/sonic_evaluation.json" \
    > /dev/null &
  evaluation_pid=$!
  while kill -0 "$evaluation_pid" 2>/dev/null; do
    echo "SONIC held-out evaluation is running (PID $evaluation_pid)."
    sleep 60
  done
  wait "$evaluation_pid"
  uv run python - "$result_dir" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = json.loads((root / "sonic_evaluation.json").read_text())
summary = {
    "task": report["task"],
    "dataset_fingerprint": report["dataset_fingerprint"],
    "robot_fingerprint": report["robot_fingerprint"],
    "release_checkpoint_sha256": report["release_checkpoint_sha256"],
    "release_artifact_sha256": report["release_artifact_sha256"],
    "evaluated_clips": len(report["per_clip"]),
    "global": report["global"],
    "acceptance": report["acceptance"],
    "reference_action_oracle": {
        "evaluated_clips": len(report["reference_action_oracle"]["per_clip"]),
        "global": report["reference_action_oracle"]["global"],
        "acceptance": report["reference_action_oracle"]["acceptance"],
    },
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
files = {}
for name in (
    "benchmark.json",
    "checkpoint_path.txt",
    "sonic_evaluation.json",
    "summary.json",
):
    files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(root / "manifest.json").write_text(
    json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True)
    + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
fi
"""


def stage_twist_inputs(
    target: tuple[str, str, int],
    selection: GitSelection,
    config: VastTrainConfig,
    *,
    executor: SshExecutor,
    rsync_executor: RsyncExecutor,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str, str]:
    """Transfer and verify explicit TWIST inputs before remote training starts."""

    dataset_path, teacher_checkpoint_path, _, _, warm_start_checkpoint_path = (
        stage_training_inputs(
            target,
            selection,
            config,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
        )
    )
    return dataset_path, teacher_checkpoint_path, warm_start_checkpoint_path


def stage_training_inputs(
    target: tuple[str, str, int],
    selection: GitSelection,
    config: VastTrainConfig,
    *,
    executor: SshExecutor,
    rsync_executor: RsyncExecutor,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str, str, str, str]:
    """Transfer and verify all explicit dataset and checkpoint inputs."""

    if not selection.dataset_files:
        return "", "", "", "", ""
    if selection.dataset_root is None:
        dataset_path = _stage_files(
            target,
            selection.dataset_files,
            remote_root=config.remote_dataset_dir,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
            expected_dataset_fingerprint=selection.dataset_fingerprint,
        )
    else:
        dataset_path = _stage_tree_files(
            target,
            selection.dataset_files,
            local_root=selection.dataset_root,
            remote_root=config.remote_sonic_dataset_dir,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
            expected_dataset_fingerprint=selection.dataset_fingerprint,
        )
    teacher_checkpoint_path = ""
    if selection.teacher_checkpoint_path is not None:
        teacher_checkpoint_path = _stage_directory(
            target,
            selection.teacher_checkpoint_path,
            remote_root=config.remote_teacher_checkpoint_dir,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
        )
    sonic_artifact_path = ""
    if selection.sonic_artifact_path is not None:
        sonic_artifact_path = _stage_directory(
            target,
            selection.sonic_artifact_path,
            remote_root=config.remote_sonic_artifact_dir,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
        )
    resume_checkpoint_path = ""
    if selection.resume_checkpoint_path is not None:
        resume_checkpoint_path = _stage_directory(
            target,
            selection.resume_checkpoint_path,
            remote_root=config.remote_sonic_resume_dir,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
        )
    warm_start_checkpoint_path = ""
    if selection.warm_start_checkpoint_path is not None:
        warm_start_checkpoint_path = _stage_directory(
            target,
            selection.warm_start_checkpoint_path,
            remote_root=config.remote_warm_start_checkpoint_dir,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
        )
    return (
        f"{dataset_path}/index.json",
        teacher_checkpoint_path,
        sonic_artifact_path,
        resume_checkpoint_path,
        warm_start_checkpoint_path,
    )


def _stage_files(
    target: tuple[str, str, int],
    paths: tuple[Path, ...],
    *,
    remote_root: str,
    timeout_seconds: int,
    retries: int,
    executor: SshExecutor,
    rsync_executor: RsyncExecutor,
    sleep: Callable[[float], None],
    expected_dataset_fingerprint: str | None = None,
) -> str:
    manifests = {path.name: _sha256(path) for path in paths}
    _prepare_remote_directory(target, remote_root, executor, timeout_seconds)
    command = _rsync_command(target, [str(path) for path in paths], remote_root)
    _run_rsync_with_retries(
        command,
        timeout_seconds=timeout_seconds,
        retries=retries,
        executor=rsync_executor,
        sleep=sleep,
    )
    _verify_remote_files(
        target,
        remote_root,
        manifests,
        executor=executor,
        timeout_seconds=timeout_seconds,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    return remote_root


def _stage_tree_files(
    target: tuple[str, str, int],
    paths: tuple[Path, ...],
    *,
    local_root: Path,
    remote_root: str,
    timeout_seconds: int,
    retries: int,
    executor: SshExecutor,
    rsync_executor: RsyncExecutor,
    sleep: Callable[[float], None],
    expected_dataset_fingerprint: str | None = None,
) -> str:
    local_root = local_root.resolve()
    relative_paths = []
    manifests = {}
    for path in paths:
        resolved = path.resolve()
        if local_root not in resolved.parents or not resolved.is_file():
            raise ValueError(f"transfer file escapes its dataset root: {path}")
        relative = resolved.relative_to(local_root).as_posix()
        relative_paths.append(relative)
        manifests[relative] = _sha256(resolved)
    _prepare_remote_directory(target, remote_root, executor, timeout_seconds)
    sources = [f"{local_root}/./{relative}" for relative in relative_paths]
    command = _rsync_command(target, sources, remote_root, relative=True)
    _run_rsync_with_retries(
        command,
        timeout_seconds=timeout_seconds,
        retries=retries,
        executor=rsync_executor,
        sleep=sleep,
    )
    _verify_remote_files(
        target,
        remote_root,
        manifests,
        executor=executor,
        timeout_seconds=timeout_seconds,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    return remote_root


def _stage_directory(
    target: tuple[str, str, int],
    source: Path,
    *,
    remote_root: str,
    timeout_seconds: int,
    retries: int,
    executor: SshExecutor,
    rsync_executor: RsyncExecutor,
    sleep: Callable[[float], None],
) -> str:
    files = tuple(sorted(path for path in source.rglob("*") if path.is_file()))
    if not files:
        raise ValueError(f"transfer source contains no files: {source}")
    manifests = {path.relative_to(source).as_posix(): _sha256(path) for path in files}
    _prepare_remote_directory(target, remote_root, executor, timeout_seconds)
    command = _rsync_command(target, [f"{source}/"], remote_root)
    _run_rsync_with_retries(
        command,
        timeout_seconds=timeout_seconds,
        retries=retries,
        executor=rsync_executor,
        sleep=sleep,
    )
    _verify_remote_files(
        target,
        remote_root,
        manifests,
        executor=executor,
        timeout_seconds=timeout_seconds,
    )
    return remote_root


def _prepare_remote_directory(
    target: tuple[str, str, int],
    remote_root: str,
    executor: SshExecutor,
    timeout_seconds: int,
) -> None:
    bootstrap = (
        "export DEBIAN_FRONTEND=noninteractive; "
        "command -v rsync >/dev/null 2>&1 || "
        "(apt-get update && apt-get install -y --no-install-recommends rsync); "
        + shlex.join(["mkdir", "-p", remote_root])
    )
    result = executor.run(
        target,
        bootstrap,
        timeout_seconds=min(300, timeout_seconds),
    )
    if result.returncode != 0:
        _raise_remote_failure(result)


def _rsync_command(
    target: tuple[str, str, int],
    sources: list[str],
    remote_root: str,
    *,
    relative: bool = False,
) -> list[str]:
    user, host, port = target
    transport = shlex.join(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"UserKnownHostsFile={EPHEMERAL_KNOWN_HOSTS_PATH}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(port),
        ]
    )
    command = [
        "rsync",
        "--archive",
        "--partial",
        "--append-verify",
        "--human-readable",
        "--info=progress2",
    ]
    if relative:
        command.append("--relative")
    command.extend(
        [
            "--rsh",
            transport,
            *sources,
            f"{user}@{host}:{remote_root}/",
        ]
    )
    return command


def _run_rsync_with_retries(
    command: list[str],
    *,
    timeout_seconds: int,
    retries: int,
    executor: RsyncExecutor,
    sleep: Callable[[float], None],
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            executor.run(command, timeout_seconds=remaining)
            return
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            last_error = exc
            if attempt < retries and time.monotonic() < deadline:
                print(f"training input transfer attempt {attempt} failed; resuming.")
                sleep(5)
    raise RuntimeError(f"training input transfer failed: {last_error}") from last_error


def _verify_remote_files(
    target: tuple[str, str, int],
    remote_root: str,
    manifests: dict[str, str],
    *,
    executor: SshExecutor,
    timeout_seconds: int,
    expected_dataset_fingerprint: str | None = None,
) -> None:
    verifier = r"""import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads(sys.argv[2])
for relative, expected in manifest.items():
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise RuntimeError(f"invalid or missing staged file: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"staged file checksum mismatch: {relative}")
fingerprint = sys.argv[3]
if fingerprint:
    index = json.loads((root / "index.json").read_text())
    if index.get("dataset_fingerprint") != fingerprint:
        raise RuntimeError("staged training dataset fingerprint mismatch")
print(f"Verified {len(manifest)} staged files under {root}.")
"""
    command = shlex.join(
        [
            "python3",
            "-c",
            verifier,
            remote_root,
            json.dumps(manifests, sort_keys=True),
            expected_dataset_fingerprint or "",
        ]
    )
    result = executor.run(
        target,
        command,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        _raise_remote_failure(result)
    _print_remote_output(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rsync_download_command(
    target: tuple[str, str, int], remote_root: str, local_root: Path
) -> list[str]:
    user, host, port = target
    transport = shlex.join(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"UserKnownHostsFile={EPHEMERAL_KNOWN_HOSTS_PATH}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(port),
        ]
    )
    return [
        "rsync",
        "--archive",
        "--partial",
        "--append-verify",
        "--human-readable",
        "--info=progress2",
        "--rsh",
        transport,
        f"{user}@{host}:{remote_root}/",
        f"{local_root}/",
    ]


def _download_sonic_results(
    target: tuple[str, str, int],
    remote_root: str,
    local_root: Path,
    *,
    expected_dataset_fingerprint: str,
    timeout_seconds: int,
    retries: int,
    executor: RsyncExecutor,
    sleep: Callable[[float], None],
) -> Path:
    local_root.mkdir(parents=True, exist_ok=True)
    command = _rsync_download_command(target, remote_root, local_root)
    _run_rsync_with_retries(
        command,
        timeout_seconds=timeout_seconds,
        retries=retries,
        executor=executor,
        sleep=sleep,
    )
    _validate_local_sonic_results(
        local_root,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    return local_root


def _validate_local_sonic_results(
    root: Path, *, expected_dataset_fingerprint: str
) -> None:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"downloaded SONIC result manifest is invalid: {exc}") from exc
    expected_names = {
        "benchmark.json",
        "checkpoint_path.txt",
        "sonic_evaluation.json",
        "summary.json",
    }
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != expected_names
    ):
        raise ValueError("downloaded SONIC result manifest differs from its schema")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"downloaded SONIC result checksum mismatch: {name}")
    report = json.loads((root / "sonic_evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    benchmark_result = benchmark.get("sonic_benchmark")
    if (
        not isinstance(benchmark_result, dict)
        or benchmark_result.get("exit_status") != 0
        or benchmark_result.get("num_envs", 0) <= 0
        or benchmark_result.get("updates") != 3
        or report.get("task") != "sonic"
        or report.get("dataset_fingerprint") != expected_dataset_fingerprint
        or summary.get("task") != "sonic"
        or summary.get("dataset_fingerprint") != expected_dataset_fingerprint
        or summary.get("evaluated_clips") != len(report.get("per_clip", ()))
        or summary.get("global") != report.get("global")
        or summary.get("acceptance") != report.get("acceptance")
        or summary.get("reference_action_oracle")
        != {
            "evaluated_clips": len(
                report.get("reference_action_oracle", {}).get("per_clip", ())
            ),
            "global": report.get("reference_action_oracle", {}).get("global"),
            "acceptance": report.get("reference_action_oracle", {}).get("acceptance"),
        }
    ):
        raise ValueError("downloaded SONIC evaluation summary differs from its report")


def run_remote_training(
    target: tuple[str, str, int],
    selection: GitSelection,
    config: VastTrainConfig,
    run_name: str,
    *,
    wandb_key: str,
    github_token: str,
    sleep: Callable[[float], None] = time.sleep,
    executor: SshExecutor | None = None,
    rsync_executor: RsyncExecutor | None = None,
    instance_api: VastApi | None = None,
    instance_id: int | None = None,
) -> None:
    if executor is None:
        executor = SshExecutor()
    if rsync_executor is None:
        rsync_executor = RsyncExecutor()
    (
        dataset_path,
        teacher_checkpoint_path,
        sonic_artifact_path,
        resume_checkpoint_path,
        warm_start_checkpoint_path,
    ) = stage_training_inputs(
        target,
        selection,
        config,
        executor=executor,
        rsync_executor=rsync_executor,
        sleep=sleep,
    )
    payload = json.dumps(
        {
            "environment": {
                "WANDB_API_KEY": wandb_key,
                "GITHUB_TOKEN": github_token,
            },
            "script": REMOTE_TRAIN_SCRIPT,
        }
    )
    job_dir = f"/workspace/.bello-vast-jobs/{run_name}"
    start_command = remote_start_command(
        job_dir,
        selection,
        config,
        run_name,
        dataset_path=dataset_path,
        teacher_checkpoint_path=teacher_checkpoint_path,
        sonic_artifact_path=sonic_artifact_path,
        resume_checkpoint_path=resume_checkpoint_path,
        warm_start_checkpoint_path=warm_start_checkpoint_path,
    )
    started_at = time.monotonic()
    ssh_start_deadline = started_at + config.ssh_timeout_seconds
    job_deadline = started_at + config.job_timeout_seconds
    disconnect_reported = False

    while True:
        now = time.monotonic()
        if now >= job_deadline:
            raise TimeoutError(
                f"remote training exceeded {config.job_timeout_seconds} seconds "
                "while starting"
            )
        if now >= ssh_start_deadline:
            raise TimeoutError("SSH did not recover while starting remote training")
        try:
            result = executor.run(
                target,
                start_command,
                input_text=payload,
                timeout_seconds=min(
                    SSH_SUBPROCESS_TIMEOUT_SECONDS,
                    job_deadline - now,
                    ssh_start_deadline - now,
                ),
            )
        except subprocess.TimeoutExpired:
            _require_recoverable_instance(instance_api, instance_id)
            if not disconnect_reported:
                print("SSH timed out while starting; retrying the detached remote job.")
                disconnect_reported = True
            sleep(5)
            continue
        if result.returncode == 0:
            _print_remote_output(result)
            break
        if result.returncode != 255:
            _raise_remote_failure(result)
        _require_recoverable_instance(instance_api, instance_id)
        if time.monotonic() >= ssh_start_deadline:
            raise TimeoutError("SSH did not recover while starting remote training")
        if not disconnect_reported:
            print("SSH disconnected while starting; retrying the detached remote job.")
            disconnect_reported = True
        sleep(5)

    offset = 0
    disconnected_at: float | None = None
    last_progress_at = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= job_deadline:
            raise TimeoutError(
                f"remote training exceeded {config.job_timeout_seconds} seconds"
            )
        if (
            disconnected_at is None
            and now - last_progress_at >= config.progress_timeout_seconds
        ):
            raise TimeoutError(
                "remote training produced no new output for "
                f"{config.progress_timeout_seconds} seconds"
            )
        call_deadlines = [job_deadline]
        if disconnected_at is None:
            call_deadlines.append(last_progress_at + config.progress_timeout_seconds)
        try:
            result = executor.run(
                target,
                remote_status_command(job_dir, offset),
                timeout_seconds=min(
                    SSH_SUBPROCESS_TIMEOUT_SECONDS,
                    min(call_deadlines) - now,
                ),
            )
        except subprocess.TimeoutExpired:
            _require_recoverable_instance(instance_api, instance_id)
            if disconnected_at is None:
                disconnected_at = time.monotonic()
                print("SSH timed out; remote training continues while reconnecting.")
            sleep(5)
            continue
        if result.returncode == 255:
            _require_recoverable_instance(instance_api, instance_id)
            if disconnected_at is None:
                disconnected_at = time.monotonic()
                print("SSH disconnected; remote training continues while reconnecting.")
            sleep(5)
            continue
        if result.returncode != 0:
            _raise_remote_failure(result)
        if disconnected_at is not None:
            print("SSH reconnected; resuming remote training output.")
            disconnected_at = None
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        update = parse_remote_job_update(result.stdout)
        if update.output:
            sys.stdout.flush()
            sys.stdout.buffer.write(update.output)
            sys.stdout.buffer.flush()
        if update.offset != offset:
            last_progress_at = time.monotonic()
        offset = update.offset
        if update.state == "running":
            sleep(10)
            continue
        if update.state == "lost":
            raise RuntimeError("detached remote training exited without a status")
        if update.returncode:
            raise subprocess.CalledProcessError(
                update.returncode,
                ["detached remote training", run_name],
            )
        break
    if config.experiment == "sonic":
        if selection.dataset_fingerprint is None:
            raise RuntimeError("SONIC remote results require a dataset fingerprint")
        local_root = Path(config.local_result_dir).expanduser().resolve() / run_name
        downloaded = _download_sonic_results(
            target,
            f"{job_dir}/results",
            local_root,
            expected_dataset_fingerprint=selection.dataset_fingerprint,
            timeout_seconds=config.transfer_timeout_seconds,
            retries=config.transfer_retries,
            executor=rsync_executor,
            sleep=sleep,
        )
        print(f"Downloaded verified SONIC evaluation results to {downloaded}.")


def _print_remote_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def _raise_remote_failure(result: subprocess.CompletedProcess[str]) -> None:
    _print_remote_output(result)
    raise subprocess.CalledProcessError(
        result.returncode,
        result.args,
        output=result.stdout,
        stderr=result.stderr,
    )


def remote_start_command(
    job_dir: str,
    selection: GitSelection,
    config: VastTrainConfig,
    run_name: str,
    *,
    dataset_path: str = "",
    teacher_checkpoint_path: str = "",
    sonic_artifact_path: str = "",
    resume_checkpoint_path: str = "",
    warm_start_checkpoint_path: str = "",
) -> str:
    bootstrap = r"""import json, os, pathlib, subprocess, sys
payload = json.load(sys.stdin)
env = payload["environment"]
script = payload["script"]
job_dir = pathlib.Path(sys.argv[1])
job_dir.mkdir(parents=True, exist_ok=True)
script_path = job_dir / "train.sh"
log_path = job_dir / "output.log"
pid_path = job_dir / "pid"
status_path = job_dir / "status"
if status_path.exists():
    print(f"Detached remote job already completed in {job_dir}.")
    raise SystemExit(0)
if pid_path.exists():
    try:
        os.kill(int(pid_path.read_text()), 0)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Detached remote job has a stale PID in {job_dir}.") from error
    else:
        print(f"Detached remote job already running in {job_dir}.")
        raise SystemExit(0)
script_path.write_text(script)
wrapper = r'''set +e
script_path="$1"
status_path="$2"
shift 2
heartbeat() {
  while sleep 300; do
    printf 'remote job heartbeat %s\n' "$(date -u +%FT%TZ)"
  done
}
heartbeat &
heartbeat_pid=$!
bash "$script_path" "$@"
status=$?
kill "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true
status_tmp="$status_path.$$"
printf '%s\n' "$status" > "$status_tmp"
mv "$status_tmp" "$status_path"
exit "$status"
'''
with log_path.open("ab", buffering=0) as log:
    process = subprocess.Popen(
        ["bash", "-c", wrapper, "bello-vast-job", str(script_path), str(status_path), *sys.argv[2:]],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ | env,
        start_new_session=True,
        close_fds=True,
    )
pid_path.write_text(f"{process.pid}\n")
print(f"Started detached remote job {process.pid} in {job_dir}.")
"""
    return shlex.join(
        [
            "python3",
            "-c",
            bootstrap,
            job_dir,
            selection.branch,
            selection.commit,
            selection.repo_url,
            str(config.remote_dir),
            config.experiment,
            str(selection.motion_path or ""),
            (
                ""
                if config.end_effector_payload_mass_kg is None
                else f"{config.end_effector_payload_mass_kg:.17g}"
            ),
            run_name,
            dataset_path,
            teacher_checkpoint_path,
            warm_start_checkpoint_path,
            str(config.allow_git_commit_mismatch).lower(),
            sonic_artifact_path,
            resume_checkpoint_path,
            str(config.total_timesteps or ""),
            str(config.num_envs or ""),
            str(config.checkpoint_every_updates or ""),
            str(config.learning_rate or ""),
            f"{job_dir}/results",
        ]
    )


def remote_status_command(job_dir: str, offset: int) -> str:
    status_reader = r"""import base64, json, os, pathlib, sys
job_dir = pathlib.Path(sys.argv[1])
offset = int(sys.argv[2])
log_path = job_dir / "output.log"
status_path = job_dir / "status"
pid_path = job_dir / "pid"
if not log_path.is_file():
    raise RuntimeError(f"detached remote job log is missing: {log_path}")
size = log_path.stat().st_size
if offset > size:
    raise RuntimeError(f"log offset {offset} exceeds current size {size}")
with log_path.open("rb") as log:
    log.seek(offset)
    output = log.read(262144)
next_offset = offset + len(output)
returncode = int(status_path.read_text()) if status_path.exists() else None
if returncode is not None and next_offset >= size:
    state = "complete"
elif returncode is not None:
    state = "running"
else:
    try:
        pid = int(pid_path.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        state = "lost"
    else:
        state = "running"
print(json.dumps({
    "state": state,
    "offset": next_offset,
    "output": base64.b64encode(output).decode("ascii"),
    "returncode": returncode,
}))
"""
    return shlex.join(["python3", "-c", status_reader, job_dir, str(offset)])


def parse_remote_job_update(value: str) -> RemoteJobUpdate:
    try:
        parsed = json.loads(value)
        state = str(parsed["state"])
        offset = int(parsed["offset"])
        output = base64.b64decode(parsed["output"], validate=True)
        raw_returncode = parsed["returncode"]
        returncode = None if raw_returncode is None else int(raw_returncode)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid detached remote job response: {value!r}") from exc
    if state not in {"running", "complete", "lost"} or offset < 0:
        raise RuntimeError(f"invalid detached remote job response: {value!r}")
    return RemoteJobUpdate(
        state=state,
        offset=offset,
        output=output,
        returncode=returncode,
    )
