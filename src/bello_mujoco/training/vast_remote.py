"""Direct-SSH command construction and detached Vast training execution."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bello_mujoco.training.vast_provider import VastApi

if TYPE_CHECKING:
    from bello_mujoco.training.vast import GitSelection, VastTrainConfig


SSH_SUBPROCESS_TIMEOUT_SECONDS = 60.0


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
total_timesteps="${13}"
learning_rate="${14}"

export DEBIAN_FRONTEND=noninteractive
for apt_source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
  [ -f "$apt_source" ] || continue
  sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
    "$apt_source"
done
apt-get update
apt-get install -y --no-install-recommends ca-certificates ffmpeg git libegl1 libgl1 libglib2.0-0 libosmesa6 libxrender1 python3 python3-pip rsync

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
  echo "bello-stack submodule is absent, uninitialized, or not at its pinned commit" >&2
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
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=5
export WARP_CACHE_PATH="$PWD/.warp-cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

uv python install
uv sync --frozen
uv run python -c 'import jax; devices=jax.devices(); print({"backend": jax.default_backend(), "devices": [str(x) for x in devices]}); assert jax.default_backend() == "gpu"'
uv run python - <<'PY'
from contextlib import ExitStack
from pathlib import Path

import mujoco

from bello_mujoco.evaluation.render import VideoWriter

output_path = Path("/tmp/bello-render-smoke.mp4")
output_path.unlink(missing_ok=True)
model = mujoco.MjModel.from_xml_string(
    '<mujoco><visual><global offwidth="64" offheight="64"/></visual>'
    '<worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
with ExitStack() as resources:
    renderer = mujoco.Renderer(model, width=64, height=64)
    resources.callback(renderer.close)
    renderer.update_scene(data)
    frame = renderer.render()
    writer = VideoWriter(output_path, width=64, height=64, fps=1)
    resources.callback(writer.close)
    writer.append_data(frame)
if frame.shape != (64, 64, 3) or not output_path.is_file():
    raise RuntimeError("headless render-and-encode preflight produced no video")
print({"render_backend": "osmesa", "frame_shape": frame.shape})
output_path.unlink()
PY

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
if [ "$allow_git_commit_mismatch" = "true" ]; then
  training_overrides+=("allow_git_commit_mismatch=true")
fi
if [ -n "$total_timesteps" ]; then
  training_overrides+=("total_timesteps=$total_timesteps")
fi
if [ -n "$learning_rate" ]; then
  training_overrides+=("learning_rate=$learning_rate")
fi

preflight_overrides=(
  "${training_overrides[@]}"
  total_timesteps=768
  num_envs=32
  wandb_enabled=false
)
if [[ "$experiment" != twist_* && "$experiment" != twist2 ]]; then
  preflight_overrides+=(video_enabled=false)
fi
if [[ "$experiment" == bello_* ]]; then
  preflight_overrides+=("checkpoint_directory=checkpoints/vast-preflight-$run_name" checkpoint_name=smoke)
fi

uv run bello-train \
  "${preflight_overrides[@]}"

if [[ "$experiment" == bello_* ]]; then
  uv run python - "checkpoints/vast-preflight-$run_name" <<'PY'
import sys
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from bello_mujoco.evaluation.render import CameraConfig, render_policy_rollout
from bello_mujoco.locomotion_runtime import load_locomotion_policy_runtime
from bello_mujoco.training.checkpoint import load_checkpoint_update

paths = list(Path(sys.argv[1]).glob("smoke_*"))
if len(paths) != 1:
    raise RuntimeError(f"expected one preflight checkpoint, got {len(paths)}")
checkpoint = paths[0]
if load_checkpoint_update(checkpoint) < 1:
    raise RuntimeError("preflight did not complete a PPO update")
runtime = load_locomotion_policy_runtime(checkpoint, use_viewer_model=True)
leaves = jax.tree_util.tree_leaves(eqx.filter(runtime.policy, eqx.is_array))
if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
    raise RuntimeError("restored preflight policy contains nonfinite parameters")
video = Path(sys.argv[1]) / "restored-policy.mp4"
render_policy_rollout(
    runtime.env, runtime.policy, output_path=video, command=(0.0, 0.0, 0.0),
    steps=25, width=256, height=192, fps=50, stop_on_done=True,
    camera_config=CameraConfig(), ppo_config=runtime.metadata.ppo,
    observation_norm_state=runtime.normalization.observation,
    obs_stack_size=runtime.metadata.ppo.policy.obs_stack_size,
    record_standing=True,
)
if not video.is_file() or video.stat().st_size == 0:
    raise RuntimeError("restored policy produced no video")
if not video.with_suffix(".jsonl").is_file():
    raise RuntimeError("restored policy produced no settling diagnostics")
print({"preflight_checkpoint_restored": str(checkpoint), "policy_video": str(video)})
PY
fi

uv run bello-train \
  "${training_overrides[@]}" \
  wandb_enabled=true \
  run_name="$run_name" \
  checkpoint_name="$run_name"
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

    if not selection.dataset_files:
        return "", "", ""
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
    target: tuple[str, str, int], sources: list[str], remote_root: str
) -> list[str]:
    user, host, port = target
    transport = shlex.join(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
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
        *sources,
        f"{user}@{host}:{remote_root}/",
    ]


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
                print(f"TWIST input transfer attempt {attempt} failed; resuming.")
                sleep(5)
    raise RuntimeError(f"TWIST input transfer failed: {last_error}") from last_error


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
        raise RuntimeError("staged TWIST dataset fingerprint mismatch")
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
) -> None:
    if executor is None:
        executor = SshExecutor()
    if rsync_executor is None:
        rsync_executor = RsyncExecutor()
    dataset_path, teacher_checkpoint_path, warm_start_checkpoint_path = (
        stage_twist_inputs(
            target,
            selection,
            config,
            executor=executor,
            rsync_executor=rsync_executor,
            sleep=sleep,
        )
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
        if time.monotonic() >= ssh_start_deadline:
            raise TimeoutError("SSH did not recover while starting remote training")
        if not disconnect_reported:
            print("SSH disconnected while starting; retrying the detached remote job.")
            disconnect_reported = True
        sleep(5)

    offset = 0
    disconnected_at: float | None = None
    last_progress_at = time.monotonic()
    lost_checks = 0
    while True:
        now = time.monotonic()
        if now >= job_deadline:
            raise TimeoutError(
                f"remote training exceeded {config.job_timeout_seconds} seconds"
            )
        if now - last_progress_at >= config.progress_timeout_seconds:
            raise TimeoutError(
                "remote training produced no new output for "
                f"{config.progress_timeout_seconds} seconds"
            )
        if (
            disconnected_at is not None
            and now - disconnected_at >= config.ssh_timeout_seconds
        ):
            raise TimeoutError(
                "SSH did not recover while detached remote training was running"
            )
        call_deadlines = [
            job_deadline,
            last_progress_at + config.progress_timeout_seconds,
        ]
        if disconnected_at is not None:
            call_deadlines.append(disconnected_at + config.ssh_timeout_seconds)
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
            if disconnected_at is None:
                disconnected_at = time.monotonic()
                print("SSH timed out; remote training continues while reconnecting.")
            sleep(5)
            continue
        if result.returncode == 255:
            now = time.monotonic()
            if disconnected_at is None:
                disconnected_at = now
                print("SSH disconnected; remote training continues while reconnecting.")
            if now - disconnected_at >= config.ssh_timeout_seconds:
                raise TimeoutError(
                    "SSH did not recover while detached remote training was running"
                )
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
            lost_checks = 0
            sleep(10)
            continue
        if update.state == "lost":
            lost_checks += 1
            print(
                f"Detached job process group missing ({lost_checks}/3); retaining rental for recheck."
            )
            if lost_checks < 3:
                sleep(10)
                continue
            try:
                diagnostics = executor.run(
                    target,
                    remote_job_diagnostics_command(job_dir),
                    timeout_seconds=max(
                        1.0, min(15.0, job_deadline - time.monotonic())
                    ),
                )
                _print_remote_output(diagnostics)
            except (subprocess.SubprocessError, OSError) as exc:
                print(
                    f"Could not capture final process diagnostics: {type(exc).__name__}"
                )
            raise RuntimeError(
                "detached remote training process group lost after three checks without a status"
            )
        if update.returncode:
            raise subprocess.CalledProcessError(
                update.returncode,
                ["detached remote training", run_name],
            )
        return


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
bash "$script_path" "$@"
status=$?
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
proc_stat = pathlib.Path(f"/proc/{process.pid}/stat")
if proc_stat.exists():
    (job_dir / "process_start").write_text(proc_stat.read_text().rsplit(") ", 1)[1].split()[19])
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
            str(config.total_timesteps or ""),
            str(config.learning_rate or ""),
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
        identity = job_dir / "process_start"
        proc_stat = pathlib.Path(f"/proc/{pid}/stat")
        if identity.exists() and proc_stat.exists():
            current_start = proc_stat.read_text().rsplit(") ", 1)[1].split()[19]
            if current_start != identity.read_text():
                raise ValueError("wrapper PID was reused")
        # Children retain the detached process group if its wrapper exits.
        os.killpg(pid, 0)
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


def remote_job_diagnostics_command(job_dir: str) -> str:
    script = r"""import pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
print("Final detached-job diagnostics (no environment or command arguments):")
for name in ("pid", "process_start", "status"):
    path = root / name
    print(name, path.read_text()[:512] if path.exists() else "missing")
for name in ("/proc/meminfo", "/sys/fs/cgroup/memory.events"):
    path = pathlib.Path(name)
    if path.exists():
        print(name, path.read_text()[:4096])
result = subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,stat,comm"], capture_output=True, text=True, timeout=5)
print(result.stdout[-16384:])
"""
    return shlex.join(["python3", "-c", script, job_dir])


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
