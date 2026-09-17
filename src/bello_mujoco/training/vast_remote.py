"""Direct-SSH command construction and detached Vast training execution."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
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

    def upload(self, target, path: Path, job_dir: str, digest: str, *, timeout_seconds):
        with path.open("rb") as source:
            return subprocess.run(
                ssh_command(target, remote_upload_command(job_dir, digest)),
                stdin=source,
                text=True,
                capture_output=True,
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


REMOTE_TRAIN_SCRIPT = r"""set -Eeuo pipefail

branch="$1"
expected_commit="$2"
repo_url="$3"
remote_dir="$4"
experiment="$5"
motion_path="$6"
end_effector_payload_mass_kg="$7"
run_name="$8"
total_timesteps="${9}"
learning_rate="${10}"

export DEBIAN_FRONTEND=noninteractive
for apt_source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
  [ -f "$apt_source" ] || continue
  sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
    "$apt_source"
done
apt-get update
apt-get install -y --no-install-recommends ca-certificates ffmpeg git libegl1 libegl-mesa0 libgl1-mesa-dri libgl1 libglib2.0-0 libxrender1 python3 python3-pip

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

export GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0
git clone --single-branch --branch "$branch" "$repo_url" "$remote_dir"
git -C "$remote_dir" submodule update --init --recursive -- deps/bello-stack
submodule_status="$(git -C "$remote_dir" submodule status --recursive -- deps/bello-stack)"
if [ -z "$submodule_status" ] || printf '%s\n' "$submodule_status" | grep -Eq '^[-+U]'; then
  echo "bello-stack submodule is absent, uninitialized, or not at its pinned commit" >&2
  exit 1
fi
unset GITHUB_TOKEN GIT_ASKPASS
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
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_DEFAULT_MATMUL_PRECISION=highest

uv python install
uv sync --frozen --group training
uv run --group training python -c 'import jax; devices=jax.devices(); print({"backend": jax.default_backend(), "devices": [str(x) for x in devices]}); assert jax.default_backend() == "gpu"'
uv run --group training python - <<'PY'
from contextlib import ExitStack
from pathlib import Path

from bello_mujoco.evaluation.software import configure_software_rendering, validate_software_renderer
configure_software_rendering()
import mujoco

from bello_mujoco.evaluation.video import VideoWriter
validate_software_renderer()

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
print({"render_backend": "llvmpipe", "frame_shape": frame.shape})
output_path.unlink()
PY

training_overrides=("experiment=$experiment")
if [ -n "$motion_path" ]; then
  training_overrides+=("motion_path=$motion_path")
fi
if [ -n "$end_effector_payload_mass_kg" ]; then
  training_overrides+=("end_effector_payload_mass_kg=$end_effector_payload_mass_kg")
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
preflight_overrides+=(video_enabled=false)
preflight_overrides+=("checkpoint_directory=checkpoints/vast-preflight-$run_name" checkpoint_name=smoke)

uv run --group training bello-train \
  "${preflight_overrides[@]}"

if [[ -n "$motion_path" ]]; then
  JAX_PLATFORMS=cpu uv run --group training python - "checkpoints/vast-preflight-$run_name" <<'PY'
import sys
from pathlib import Path

from bello_mujoco.evaluation.software import configure_software_rendering
configure_software_rendering()

import equinox as eqx
import jax
import numpy as np

from bello_mujoco.deployment import ExportConfig, export_checkpoint
from bello_mujoco.evaluation.motion import render_motion_policy
from bello_mujoco.motion.runtime import load_motion_policy_runtime
from bello_mujoco.training.checkpoint import load_checkpoint_update

directory = Path(sys.argv[1])
paths = list(directory.glob("smoke_*"))
if len(paths) != 1 or load_checkpoint_update(paths[0]) < 1:
    raise RuntimeError("motion preflight did not save exactly one updated checkpoint")
checkpoint = paths[0]
runtime = load_motion_policy_runtime(checkpoint, use_viewer_model=True)
leaves = jax.tree_util.tree_leaves(eqx.filter(runtime.policy, eqx.is_array))
if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
    raise RuntimeError("restored motion policy contains nonfinite parameters")
video = directory / "restored-policy.mp4"
render_motion_policy(
    runtime.env, runtime.policy, video, steps=25, width=256, height=192, fps=50,
    obs_stack_size=runtime.metadata.ppo.policy.obs_stack_size,
    ppo_config=runtime.metadata.ppo,
    observation_norm_state=runtime.normalization.observation,
)
if not video.is_file() or video.stat().st_size == 0:
    raise RuntimeError("restored motion policy produced no video")
bundle = export_checkpoint(ExportConfig(
    checkpoint_path=checkpoint, output_path=directory / "policy.zip",
))
print({"motion_preflight_checkpoint_restored": str(checkpoint),
       "policy_video": str(video), "deployment_bundle": str(bundle)})
PY
fi

if [[ -z "$motion_path" && "$experiment" == bello_* ]]; then
  uv run --group training python - "checkpoints/vast-preflight-$run_name" <<'PY'
import sys
from pathlib import Path

from bello_mujoco.evaluation.software import configure_software_rendering
configure_software_rendering()

import equinox as eqx
import jax
import numpy as np

from bello_mujoco.evaluation.video import CameraConfig
from bello_mujoco.evaluation.render import render_policy_rollout
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

uv run --group training bello-train \
  "${training_overrides[@]}" \
  wandb_enabled=true \
  run_name="$run_name" \
  checkpoint_name="$run_name"
"""


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
) -> None:
    if executor is None:
        executor = SshExecutor()
    motion_path = (
        config.motion_path.expanduser().resolve() if config.motion_path else None
    )
    digest = None
    if motion_path is not None:
        with motion_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    payload = json.dumps(
        {
            "environment": {
                "WANDB_API_KEY": wandb_key,
                "GITHUB_TOKEN": github_token,
            },
            "script": REMOTE_TRAIN_SCRIPT,
            "timeout_seconds": config.job_timeout_seconds,
            "motion_sha256": digest,
        }
    )
    job_dir = f"/workspace/.bello-vast/{run_name}"
    start_command = remote_start_command(
        job_dir,
        selection,
        config,
        run_name,
    )
    started_at = time.monotonic()
    ssh_start_deadline = started_at + config.ssh_timeout_seconds
    job_deadline = started_at + config.job_timeout_seconds
    disconnect_reported = False
    uploaded = motion_path is None

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
            if not uploaded:
                result = executor.upload(
                    target,
                    motion_path,
                    job_dir,
                    digest,
                    timeout_seconds=min(ssh_start_deadline, job_deadline) - now,
                )
                if result.returncode == 0:
                    uploaded = True
                    print("Motion uploaded and SHA-256 verified.")
                    continue
                if result.returncode != 255:
                    _raise_remote_failure(result)
                print("Motion upload disconnected; retrying before training starts.")
                sleep(5)
                continue
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
            sleep(10)
            continue
        if update.state == "lost":
            raise RuntimeError("detached remote training exited without a status")
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


def remote_upload_command(job_dir: str, digest: str) -> str:
    receiver = r"""import fcntl, hashlib, os, pathlib, sys, tempfile
job_dir = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
root = job_dir.parent
if not (root / "armed").is_file():
    raise RuntimeError("Vast host teardown supervisor is not armed")
os.kill(int((root / "supervisor_pid").read_text()), 0)
job_dir.mkdir(parents=True, exist_ok=True)
with (root / "launch.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    destination = job_dir / "motion.npz"
    with tempfile.TemporaryDirectory(prefix=".upload-", dir=job_dir) as staging:
        path = pathlib.Path(staging) / "motion.npz"
        digest = hashlib.sha256()
        with path.open("wb") as output:
            while chunk := sys.stdin.buffer.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected:
            raise RuntimeError("motion upload SHA-256 mismatch; training was not started")
        if (job_dir / "pid").exists() or (job_dir / "status").exists():
            existing_digest = hashlib.sha256()
            with destination.open("rb") as existing:
                while chunk := existing.read(1024 * 1024):
                    existing_digest.update(chunk)
            if existing_digest.hexdigest() != expected:
                raise RuntimeError("cannot replace motion for a started job")
        else:
            path.replace(destination)
print("Motion upload verified")
"""
    return shlex.join(["python3", "-c", receiver, job_dir, digest])


def remote_start_command(
    job_dir: str,
    selection: GitSelection,
    config: VastTrainConfig,
    run_name: str,
) -> str:
    bootstrap = r"""import fcntl, hashlib, json, os, pathlib, subprocess, sys
payload = json.load(sys.stdin)
env = payload["environment"]
script = payload["script"]
job_dir = pathlib.Path(sys.argv[1])
supervisor_root = job_dir.parent
if not (supervisor_root / "armed").is_file():
    raise RuntimeError("Vast host teardown supervisor is not armed")
os.kill(int((supervisor_root / "supervisor_pid").read_text()), 0)
job_dir.mkdir(parents=True, exist_ok=True)
launch_lock = (supervisor_root / "launch.lock").open("a")
fcntl.flock(launch_lock, fcntl.LOCK_EX)
if sys.argv[7]:
    digest = hashlib.sha256()
    with (job_dir / "motion.npz").open("rb") as motion:
        while chunk := motion.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != payload["motion_sha256"]:
        raise RuntimeError("uploaded motion changed; refusing to start training")
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
supervisor_root="$3"
timeout_seconds="$4"
shift 4
timeout --signal=TERM --kill-after=30s "$timeout_seconds" bash "$script_path" "$@"
status=$?
status_tmp="$status_path.$$"
printf '%s\n' "$status" > "$status_tmp"
mv "$status_tmp" "$status_path"
cp "$status_path" "$supervisor_root/status.tmp"
mv "$supervisor_root/status.tmp" "$supervisor_root/status"
exit "$status"
'''
with log_path.open("ab", buffering=0) as log:
    process = subprocess.Popen(
        ["bash", "-c", wrapper, "bello-vast-job", str(script_path), str(status_path), str(supervisor_root), str(payload["timeout_seconds"]), *sys.argv[2:]],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ | env,
        start_new_session=True,
        close_fds=True,
    )
pid_tmp = job_dir / "pid.tmp"
pid_tmp.write_text(f"{process.pid}\n")
pid_tmp.replace(pid_path)
supervisor_pid_path = supervisor_root / "job_pid.tmp"
supervisor_pid_path.write_text(str(process.pid))
supervisor_pid_path.replace(supervisor_root / "job_pid")
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
            str(Path(job_dir) / "motion.npz") if config.motion_path is not None else "",
            (
                ""
                if config.end_effector_payload_mass_kg is None
                else f"{config.end_effector_payload_mass_kg:.17g}"
            ),
            run_name,
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
