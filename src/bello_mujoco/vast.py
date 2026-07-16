"""Vast.ai provisioning and remote training orchestration."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import Any, Protocol
from urllib.parse import urlparse

import keyring


VAST_KEYRING_SERVICE = "vast"
VAST_KEYRING_USERNAME = "api_key"
WANDB_KEYRING_SERVICE = "wandb"
WANDB_KEYRING_USERNAME = "api_key"
GITHUB_KEYRING_SERVICE = "github"
GITHUB_KEYRING_USERNAME = "bello_mujoco_read_token"


@dataclass(frozen=True)
class VastOfferConfig:
    """Hardware and marketplace constraints for a training instance."""

    gpu_name: str = "RTX_5090"
    num_gpus: int = 1
    min_cpu_ram_gb: int = 64
    disk_gb: int = 80
    min_reliability: float = 0.98
    min_cuda: float = 13.0
    image: str = "vastai/base-image:cuda-13.0.3-auto"
    search_limit: int = 10
    korea_countries: tuple[str, ...] = ("KR",)
    asia_countries: tuple[str, ...] = (
        "JP",
        "SG",
        "TW",
        "HK",
        "CN",
        "VN",
        "TH",
        "MY",
        "ID",
        "PH",
        "IN",
    )


@dataclass(frozen=True)
class VastTrainConfig:
    """User-facing configuration for a complete Vast training lifecycle."""

    train_config: Path | None = None
    max_price: float | None = None
    branch: str | None = None
    run_name: str | None = None
    yes: bool = False
    dry_run: bool = False
    remote_dir: str = "/workspace/bello_mujoco"
    ready_timeout_seconds: int = 900
    ssh_timeout_seconds: int = 900
    destroy_retries: int = 3
    offer: VastOfferConfig = field(default_factory=VastOfferConfig)
    hydra_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitSelection:
    branch: str
    commit: str
    config_path: PurePosixPath
    repo_url: str


@dataclass(frozen=True)
class VastOffer:
    id: int
    gpu_name: str
    dph_total: float
    dlperf_per_dphtotal: float
    reliability: float
    geolocation: str
    cpu_ram_mb: int
    gpu_ram_mb: int
    num_gpus: int
    cuda_max_good: float
    disk_space_gb: float
    direct_port_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VastOffer:
        offer_id = value.get("id", value.get("ask_contract_id"))
        if offer_id is None:
            raise ValueError(f"Vast offer has no id: {value!r}")
        return cls(
            id=int(offer_id),
            gpu_name=str(value.get("gpu_name", "unknown")),
            dph_total=float(value.get("dph_total", float("inf"))),
            dlperf_per_dphtotal=float(
                value.get("dlperf_per_dphtotal", value.get("dlperf_usd", 0.0))
            ),
            reliability=float(value.get("reliability", 0.0)),
            geolocation=str(value.get("geolocation", "unknown")),
            cpu_ram_mb=int(value.get("cpu_ram", 0)),
            gpu_ram_mb=int(value.get("gpu_ram", 0)),
            num_gpus=int(value.get("num_gpus", 0)),
            cuda_max_good=float(value.get("cuda_max_good", 0.0)),
            disk_space_gb=float(value.get("disk_space", 0.0)),
            direct_port_count=int(value.get("direct_port_count", 0)),
        )


class VastApi(Protocol):
    def search_offers(
        self, query: str, *, order: str, limit: int, storage: float
    ) -> list[Mapping[str, Any]]: ...

    def create_instance(self, offer_id: int, **kwargs: Any) -> int: ...

    def show_instance(self, instance_id: int) -> Mapping[str, Any]: ...

    def show_instances(self) -> list[Mapping[str, Any]]: ...

    def ssh_url(self, instance_id: int) -> str: ...

    def destroy_instance(self, instance_id: int) -> None: ...


class VastSdkClient:
    """Small typed boundary around the Vast SDK's mixed response shapes."""

    def __init__(self, api_key: str):
        from vastai import VastAI

        self._client = VastAI(api_key=api_key, raw=True)

    def search_offers(
        self, query: str, *, order: str, limit: int, storage: float
    ) -> list[Mapping[str, Any]]:
        value = _json_value(
            self._client.search_offers(
                query=query,
                order=order,
                limit=limit,
                storage=storage,
            )
        )
        if isinstance(value, Mapping):
            value = value.get("offers", value.get("instances", []))
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            raise RuntimeError(f"unexpected Vast offer response: {value!r}")
        return [item for item in value if isinstance(item, Mapping)]

    def create_instance(self, offer_id: int, **kwargs: Any) -> int:
        value = _json_value(self._client.create_instance(id=offer_id, **kwargs))
        if isinstance(value, Mapping):
            instance_id = value.get("new_contract", value.get("id"))
            if instance_id is not None:
                return int(instance_id)
        raise RuntimeError(f"Vast did not return a new instance id: {value!r}")

    def show_instance(self, instance_id: int) -> Mapping[str, Any]:
        value = _json_value(self._client.show_instance(id=instance_id))
        if isinstance(value, Mapping) and "instances" in value:
            value = value["instances"]
        if isinstance(value, list):
            value = value[0] if value else {}
        if not isinstance(value, Mapping):
            raise RuntimeError(f"unexpected Vast instance response: {value!r}")
        return value

    def show_instances(self) -> list[Mapping[str, Any]]:
        value = _json_value(self._client.show_instances())
        if isinstance(value, Mapping):
            value = value.get("instances", [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    def ssh_url(self, instance_id: int) -> str:
        value = _json_value(self._client.ssh_url(id=instance_id))
        if isinstance(value, Mapping):
            value = value.get("ssh_url", value.get("url"))
        if not isinstance(value, str):
            raise RuntimeError(f"unexpected Vast SSH URL response: {value!r}")
        return value.strip()

    def destroy_instance(self, instance_id: int) -> None:
        self._client.destroy_instance(id=instance_id)


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def credential(env_name: str, service: str, username: str, *, required: bool) -> str:
    """Read a secret from an explicit environment override or Python keyring."""

    value = os.environ.get(env_name, "").strip()
    if not value:
        try:
            value = (keyring.get_password(service, username) or "").strip()
        except Exception as exc:
            if not required:
                return ""
            raise RuntimeError(
                f"could not read {service}/{username} from Python keyring: {exc}"
            ) from exc
    if required and not value:
        raise RuntimeError(
            f"missing {env_name}; store it with: keyring set {service} {username}"
        )
    return value


def validate_config(config: VastTrainConfig) -> None:
    if config.train_config is None:
        raise ValueError("--train-config is required")
    if config.max_price is None or config.max_price <= 0:
        raise ValueError("--max-price must be positive")
    if config.ready_timeout_seconds <= 0 or config.ssh_timeout_seconds <= 0:
        raise ValueError("readiness timeouts must be positive")
    if config.destroy_retries <= 0:
        raise ValueError("destroy-retries must be positive")
    remote_dir = PurePosixPath(config.remote_dir)
    if (
        not remote_dir.is_absolute()
        or len(remote_dir.parts) < 3
        or remote_dir.parts[1] != "workspace"
    ):
        raise ValueError("--remote-dir must be a child of /workspace")
    offer = config.offer
    if offer.num_gpus <= 0 or offer.min_cpu_ram_gb <= 0 or offer.disk_gb <= 0:
        raise ValueError("GPU count, CPU RAM, and disk size must be positive")
    if not 0.0 < offer.min_reliability <= 1.0:
        raise ValueError("offer.min-reliability must be in (0, 1]")
    if offer.search_limit <= 0:
        raise ValueError("offer.search-limit must be positive")
    for country in (*offer.korea_countries, *offer.asia_countries):
        if not re.fullmatch(r"[A-Z]{2}", country):
            raise ValueError(f"invalid two-letter country code: {country!r}")


def run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def remote_branches(repo_root: Path) -> list[str]:
    run_git(repo_root, "fetch", "--prune", "origin")
    output = run_git(
        repo_root,
        "for-each-ref",
        "--format=%(refname:strip=3)",
        "refs/remotes/origin",
    )
    return sorted(
        branch for branch in output.splitlines() if branch and branch != "HEAD"
    )


def select_branch(branches: Sequence[str]) -> str:
    if not branches:
        raise RuntimeError("origin has no remote branches")
    if not sys.stdin.isatty():
        raise RuntimeError("--branch is required when stdin is not interactive")
    print("Available origin branches:")
    for index, branch in enumerate(branches, start=1):
        print(f"  {index:2d}. {branch}")
    raw = input("Select branch number: ").strip()
    try:
        selected = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid branch selection: {raw!r}") from exc
    if selected < 1 or selected > len(branches):
        raise ValueError(f"branch selection must be between 1 and {len(branches)}")
    return branches[selected - 1]


def resolve_git_selection(repo_root: Path, config: VastTrainConfig) -> GitSelection:
    branches = remote_branches(repo_root)
    branch = config.branch or select_branch(branches)
    if branch not in branches:
        raise ValueError(f"branch {branch!r} does not exist on origin")

    config_path = PurePosixPath(str(config.train_config))
    if config_path.is_absolute() or ".." in config_path.parts:
        raise ValueError("--train-config must be a repository-relative path")
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("--train-config must name a YAML file")

    ref = f"refs/remotes/origin/{branch}"
    commit = run_git(repo_root, "rev-parse", f"{ref}^{{commit}}")
    for path in (str(config_path), "pyproject.toml", "uv.lock"):
        try:
            run_git(repo_root, "cat-file", "-e", f"{commit}:{path}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{path!r} does not exist on origin/{branch}") from exc
    repo_url = normalized_repo_url(run_git(repo_root, "remote", "get-url", "origin"))
    return GitSelection(
        branch=branch,
        commit=commit,
        config_path=config_path,
        repo_url=repo_url,
    )


def normalized_repo_url(repo_url: str) -> str:
    if repo_url.startswith("git@github.com:"):
        return f"https://github.com/{repo_url.removeprefix('git@github.com:')}"
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "origin must be an HTTPS Git remote or git@github.com URL for remote cloning"
        )
    if parsed.username or parsed.password:
        raise ValueError(
            "origin URL must not embed credentials; use the GitHub keyring"
        )
    return repo_url


def offer_query(config: VastTrainConfig, *, countries: tuple[str, ...] | None) -> str:
    offer = config.offer
    parts = [
        f"gpu_name={offer.gpu_name}",
        f"num_gpus={offer.num_gpus}",
        "verified=true",
        "rentable=true",
        "rented=false",
        "direct_port_count>=1",
        f"reliability>={offer.min_reliability}",
        f"cpu_ram>={offer.min_cpu_ram_gb}",
        f"disk_space>={offer.disk_gb}",
        f"cuda_max_good>={offer.min_cuda}",
        f"dph_total<={config.max_price}",
    ]
    if countries:
        parts.append(f"geolocation in [{','.join(countries)}]")
    return " ".join(parts)


def find_offers(api: VastApi, config: VastTrainConfig) -> tuple[str, list[VastOffer]]:
    tiers: tuple[tuple[str, tuple[str, ...] | None], ...] = (
        ("Korea", config.offer.korea_countries),
        ("Asia", config.offer.asia_countries),
        ("global", None),
    )
    for tier_name, countries in tiers:
        raw_offers = api.search_offers(
            offer_query(config, countries=countries),
            order="dlperf_usd-",
            limit=config.offer.search_limit,
            storage=float(config.offer.disk_gb),
        )
        offers = [VastOffer.from_mapping(item) for item in raw_offers]
        offers = [offer for offer in offers if offer_matches(offer, config)]
        offers.sort(key=lambda offer: (-offer.dlperf_per_dphtotal, offer.dph_total))
        if offers:
            return tier_name, offers
    raise RuntimeError(
        "no Vast offers matched the configured hardware and price limits"
    )


def offer_matches(offer: VastOffer, config: VastTrainConfig) -> bool:
    desired = config.offer
    normalized_actual = offer.gpu_name.upper().replace(" ", "_")
    normalized_desired = desired.gpu_name.upper().replace(" ", "_")
    return (
        offer.dph_total <= config.max_price
        and normalized_actual == normalized_desired
        and offer.num_gpus == desired.num_gpus
        and offer.reliability >= desired.min_reliability
        and offer.cpu_ram_mb >= desired.min_cpu_ram_gb * 1000
        and offer.cuda_max_good >= desired.min_cuda
        and offer.disk_space_gb >= desired.disk_gb
        and offer.direct_port_count >= 1
    )


def print_offers(tier: str, offers: Sequence[VastOffer], *, limit: int = 5) -> None:
    print(f"Matching {tier} offers (best value first):")
    print("  offer       $/hr   reliability  location             CPU RAM  GPU RAM")
    for offer in offers[:limit]:
        print(
            f"  {offer.id:<10d}  {offer.dph_total:>5.3f}  "
            f"{offer.reliability:>10.3f}  {offer.geolocation[:20]:<20} "
            f"{offer.cpu_ram_mb / 1024:>6.1f}G  {offer.gpu_ram_mb / 1024:>6.1f}G"
        )


def confirm_rental(offer: VastOffer) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("--yes is required when stdin is not interactive")
    answer = input(
        f"Rent offer {offer.id} ({offer.gpu_name}, ${offer.dph_total:.3f}/hr)? [y/N] "
    )
    if answer.strip().lower() not in {"y", "yes"}:
        raise RuntimeError("rental cancelled")


def generated_run_name(selection: GitSelection, config: VastTrainConfig) -> str:
    if config.run_name:
        name = config.run_name
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        name = (
            f"{selection.config_path.stem}-{selection.branch}-"
            f"{selection.commit[:8]}-{timestamp}"
        )
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    if not name:
        raise ValueError("run name is empty after sanitization")
    return name[:128]


@contextmanager
def pipeline_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "pipeline.lock"
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another bello-vast-train process is active") from exc
        yield


def write_active_state(state_path: Path, instance_id: int, label: str) -> None:
    state_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "label": label,
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def read_active_state(state_path: Path) -> tuple[int, str] | None:
    if not state_path.exists():
        return None
    value = json.loads(state_path.read_text())
    return int(value["instance_id"]), str(value["label"])


def destroy_with_retries(
    api: VastApi,
    instance_id: int,
    *,
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    def is_destroyed() -> bool:
        try:
            info = api.show_instance(instance_id)
        except Exception:
            return True
        return info.get("cur_state") == "unloaded" or info.get("actual_status") in {
            None,
            "destroyed",
        }

    for attempt in range(1, retries + 1):
        try:
            api.destroy_instance(instance_id)
            if is_destroyed():
                return True
        except Exception:
            if is_destroyed():
                return True
            if attempt == retries:
                return False
        sleep(min(2**attempt, 10))
    return False


def reconcile_stale_state(api: VastApi, state_path: Path, retries: int) -> None:
    active = read_active_state(state_path)
    if active is None:
        return
    instance_id, label = active
    matching_instances = [
        item for item in api.show_instances() if int(item.get("id", -1)) == instance_id
    ]
    if not matching_instances:
        state_path.unlink(missing_ok=True)
        return
    actual_label = matching_instances[0].get("label")
    if actual_label != label:
        raise RuntimeError(
            f"stale state label {label!r} does not match instance {instance_id} "
            f"label {actual_label!r}; refusing destructive cleanup"
        )
    print(f"Cleaning stale pipeline instance {instance_id} ({label}) before launch.")
    if not destroy_with_retries(api, instance_id, retries=retries):
        raise RuntimeError(
            f"could not destroy stale instance {instance_id}; refusing another rental"
        )
    state_path.unlink(missing_ok=True)


def recover_labeled_instance(api: VastApi, label: str) -> int | None:
    matches = [item for item in api.show_instances() if item.get("label") == label]
    if len(matches) != 1:
        return None
    value = matches[0].get("id")
    return int(value) if value is not None else None


def wait_for_instance(
    api: VastApi,
    instance_id: int,
    timeout_seconds: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        info = api.show_instance(instance_id)
        status = info.get("actual_status")
        print(f"Vast instance {instance_id}: {status}")
        if status == "running":
            return
        if status in {"exited", "offline"}:
            raise RuntimeError(f"instance entered non-recoverable state {status!r}")
        sleep(10)
    raise TimeoutError(f"instance {instance_id} did not become ready in time")


def parse_ssh_url(value: str) -> tuple[str, str, int]:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme != "ssh"
        or not parsed.username
        or not parsed.hostname
        or not parsed.port
    ):
        raise ValueError(f"invalid Vast SSH URL: {value!r}")
    return parsed.username, parsed.hostname, parsed.port


def wait_for_ssh(
    api: VastApi,
    instance_id: int,
    timeout_seconds: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str, int]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            target = parse_ssh_url(api.ssh_url(instance_id))
            user, host, port = target
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    "ConnectTimeout=10",
                    "-p",
                    str(port),
                    f"{user}@{host}",
                    "true",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return target
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            last_error = exc
            sleep(10)
    raise TimeoutError(f"SSH did not become ready: {last_error}")


REMOTE_TRAIN_SCRIPT = r"""set -Eeuo pipefail

branch="$1"
expected_commit="$2"
repo_url="$3"
remote_dir="$4"
config_path="$5"
run_name="$6"

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libegl1 libgl1 libglib2.0-0 libxrender1 python3 python3-pip
fi

if ! command -v uv >/dev/null 2>&1; then
  timeout 120 sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --break-system-packages --upgrade uv
fi
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
if [ -n "${GITHUB_TOKEN:-}" ]; then
  GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git clone --single-branch --branch "$branch" "$repo_url" "$remote_dir"
else
  GIT_TERMINAL_PROMPT=0 git clone --single-branch --branch "$branch" "$repo_url" "$remote_dir"
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv python install 3.12
uv sync --frozen
uv run python -c 'import jax; devices=jax.devices(); print({"backend": jax.default_backend(), "devices": [str(x) for x in devices]}); assert jax.default_backend() == "gpu"'

config_dir="$remote_dir/$(dirname "$config_path")"
config_name="$(basename "$config_path")"
config_name="${config_name%.*}"

uv run bello-train-ppo \
  --config-dir "$config_dir" \
  --config-name "$config_name" \
  dry_run=true \
  config.num_envs=1 \
  config.num_steps=1 \
  config.num_minibatches=1 \
  wandb_config.enabled=false

mapfile -t user_overrides < <(python3 -c 'import json, os, sys; sys.stdout.write("\n".join(json.loads(os.environ["BELLO_HYDRA_OVERRIDES"])))')
entity_override=()
if [ -n "${WANDB_ENTITY:-}" ]; then
  entity_override+=("wandb_config.entity=$WANDB_ENTITY")
fi
uv run bello-train-ppo \
  --config-dir "$config_dir" \
  --config-name "$config_name" \
  "${user_overrides[@]}" \
  dry_run=false \
  wandb_config.enabled=true \
  wandb_config.mode=online \
  wandb_config.project="$WANDB_PROJECT" \
  "${entity_override[@]}" \
  wandb_config.run_name="$run_name" \
  checkpoint_config.enabled=true \
  checkpoint_config.name="$run_name" \
  checkpoint_config.upload_wandb_artifact=true \
  checkpoint_config.wandb_artifact_name="$run_name"
"""


REMOTE_JOB_RUNNER = r"""import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

env_path = Path(sys.argv[1])
script_path = Path(sys.argv[2])
status_path = Path(sys.argv[3])
arguments = json.loads(sys.argv[4])
environment = os.environ.copy()
environment.update(json.loads(env_path.read_text()))
env_path.unlink(missing_ok=True)
try:
    return_code = subprocess.run(
        ["bash", str(script_path), *arguments],
        env=environment,
    ).returncode
except BaseException:
    traceback.print_exc()
    return_code = 127
temporary_status = status_path.with_suffix(".tmp")
temporary_status.write_text(f"{return_code}\n")
temporary_status.replace(status_path)
raise SystemExit(return_code)
"""


REMOTE_JOB_BOOTSTRAP = r"""import json
import os
from pathlib import Path
import subprocess
import sys

job_dir = Path(sys.argv[1])
arguments = sys.argv[2:]
environment = json.loads(sys.stdin.readline())
script = sys.stdin.read()
job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
for child in job_dir.iterdir():
    if child.is_file():
        child.unlink()
env_path = job_dir / "environment.json"
script_path = job_dir / "train.sh"
status_path = job_dir / "status"
log_path = job_dir / "train.log"
pid_path = job_dir / "pid"
env_path.write_text(json.dumps(environment))
env_path.chmod(0o600)
script_path.write_text(script)
script_path.chmod(0o700)
with log_path.open("ab", buffering=0) as log_file:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            os.environ["BELLO_REMOTE_JOB_RUNNER"],
            str(env_path),
            str(script_path),
            str(status_path),
            json.dumps(arguments),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pid_path.write_text(f"{process.pid}\n")
print(job_dir)
"""


def _ssh_command(target: tuple[str, str, int], remote_command: str) -> list[str]:
    user, host, port = target
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(port),
        f"{user}@{host}",
        remote_command,
    ]


def _remote_job_poll_command(job_dir: str, log_offset: int) -> str:
    quoted_dir = shlex.quote(job_dir)
    return f"""job_dir={quoted_dir}
log_size=$(stat -c %s "$job_dir/train.log" 2>/dev/null || printf 0)
if [ -f "$job_dir/status" ]; then
  state="done:$(cat "$job_dir/status")"
elif [ -f "$job_dir/pid" ] && kill -0 "$(cat "$job_dir/pid")" 2>/dev/null; then
  state=running
else
  state=lost
fi
printf 'BELLO_JOB_STATE=%s\\n' "$state"
printf 'BELLO_LOG_SIZE=%s\\n' "$log_size"
if [ "$log_size" -gt {log_offset} ]; then
  tail -c +$(({log_offset} + 1)) "$job_dir/train.log" | head -c $((log_size - {log_offset}))
fi"""


def _parse_remote_job_poll(output: str) -> tuple[str, int, str]:
    lines = output.split("\n", 2)
    if len(lines) < 3 or not lines[0].startswith("BELLO_JOB_STATE="):
        raise RuntimeError(f"invalid remote job status response: {output!r}")
    if not lines[1].startswith("BELLO_LOG_SIZE="):
        raise RuntimeError(f"invalid remote job log response: {output!r}")
    return (
        lines[0].removeprefix("BELLO_JOB_STATE="),
        int(lines[1].removeprefix("BELLO_LOG_SIZE=")),
        lines[2],
    )


def run_remote_training(
    target: tuple[str, str, int],
    selection: GitSelection,
    config: VastTrainConfig,
    run_name: str,
    *,
    wandb_key: str,
    github_token: str,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 30.0,
) -> None:
    project = os.environ.get("WANDB_PROJECT", "bello-mujoco")
    entity = os.environ.get("WANDB_ENTITY", "")
    remote_env = {
        "WANDB_API_KEY": wandb_key,
        "WANDB_PROJECT": project,
        "WANDB_ENTITY": entity,
        "GITHUB_TOKEN": github_token,
        "BELLO_HYDRA_OVERRIDES": json.dumps(config.hydra_overrides),
    }
    job_token = hashlib.sha256(f"{run_name}:{selection.commit}".encode()).hexdigest()[
        :16
    ]
    job_dir = f"/tmp/bello-vast-{job_token}"
    remote_command = shlex.join(
        [
            "env",
            f"BELLO_REMOTE_JOB_RUNNER={REMOTE_JOB_RUNNER}",
            "python3",
            "-c",
            REMOTE_JOB_BOOTSTRAP,
            job_dir,
            selection.branch,
            selection.commit,
            selection.repo_url,
            str(config.remote_dir),
            str(selection.config_path),
            run_name,
        ]
    )
    payload = json.dumps(remote_env) + "\n" + REMOTE_TRAIN_SCRIPT
    try:
        subprocess.run(
            _ssh_command(target, remote_command),
            input=payload,
            text=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 255:
            raise
        print(
            "SSH disconnected while launching the remote job; checking its "
            "detached status before deciding whether launch failed.",
            flush=True,
        )
    print(f"Started detached remote training job at {job_dir}.")

    log_offset = 0
    disconnect_deadline = time.monotonic() + config.ssh_timeout_seconds
    while True:
        try:
            result = subprocess.run(
                _ssh_command(
                    target,
                    _remote_job_poll_command(job_dir, log_offset),
                ),
                text=True,
                check=True,
                capture_output=True,
                timeout=60,
            )
            state, log_offset, new_log = _parse_remote_job_poll(result.stdout)
            if new_log:
                print(new_log, end="" if new_log.endswith("\n") else "\n", flush=True)
            disconnect_deadline = time.monotonic() + config.ssh_timeout_seconds
            if state.startswith("done:"):
                return_code = int(state.removeprefix("done:").strip())
                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code,
                        ["remote-training-job", run_name],
                    )
                return
            if state == "lost":
                raise RuntimeError(
                    f"remote training job {job_dir} disappeared without a status"
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            if time.monotonic() >= disconnect_deadline:
                raise TimeoutError(
                    "could not reconnect to the detached remote training job "
                    f"within {config.ssh_timeout_seconds} seconds"
                ) from exc
            print(f"Remote training poll failed ({exc}); retrying.", flush=True)
        except subprocess.CalledProcessError as exc:
            if exc.cmd and exc.cmd[0] == "remote-training-job":
                raise
            if time.monotonic() >= disconnect_deadline:
                raise TimeoutError(
                    "could not reconnect to the detached remote training job "
                    f"within {config.ssh_timeout_seconds} seconds"
                ) from exc
            print(
                f"Remote training SSH poll exited {exc.returncode}; retrying.",
                flush=True,
            )
        sleep(poll_interval_seconds)


class PipelineInterrupted(RuntimeError):
    pass


@contextmanager
def termination_handler():
    previous = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        raise PipelineInterrupted("received SIGTERM")

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def launch_pipeline(
    repo_root: Path,
    config: VastTrainConfig,
    *,
    api_factory: Callable[[str], VastApi] = VastSdkClient,
    remote_runner: Callable[..., None] = run_remote_training,
) -> int:
    validate_config(config)
    selection = resolve_git_selection(repo_root, config)
    run_name = generated_run_name(selection, config)
    print(
        f"Selected origin/{selection.branch} at {selection.commit[:12]} with "
        f"{selection.config_path} (run {run_name})."
    )

    if config.dry_run:
        for tier, countries in (
            ("Korea", config.offer.korea_countries),
            ("Asia", config.offer.asia_countries),
            ("global", None),
        ):
            print(f"{tier} query: {offer_query(config, countries=countries)}")
        print("Dry run: no credentials were read and no instance was rented.")
        return 0

    vast_key = credential(
        "VAST_API_KEY", VAST_KEYRING_SERVICE, VAST_KEYRING_USERNAME, required=True
    )
    wandb_key = credential(
        "WANDB_API_KEY", WANDB_KEYRING_SERVICE, WANDB_KEYRING_USERNAME, required=True
    )
    github_token = credential(
        "GITHUB_TOKEN",
        GITHUB_KEYRING_SERVICE,
        GITHUB_KEYRING_USERNAME,
        required=False,
    )
    api = api_factory(vast_key)
    state_dir = repo_root / ".vast"
    state_path = state_dir / "active_instance.json"
    label = f"bello-{run_name}"

    with pipeline_lock(state_dir), termination_handler():
        reconcile_stale_state(api, state_path, config.destroy_retries)
        tier, offers = find_offers(api, config)
        print_offers(tier, offers)
        selected_offer = offers[0]
        if not config.yes:
            confirm_rental(selected_offer)

        instance_id: int | None = None
        failure: BaseException | None = None
        try:
            try:
                instance_id = api.create_instance(
                    selected_offer.id,
                    image=config.offer.image,
                    disk=float(config.offer.disk_gb),
                    label=label,
                    runtype="ssh_direc ssh_proxy",
                    cancel_unavail=True,
                )
            except BaseException:
                instance_id = recover_labeled_instance(api, label)
                if instance_id is None:
                    raise
            write_active_state(state_path, instance_id, label)
            print(f"Created Vast instance {instance_id}; teardown is armed.")
            wait_for_instance(api, instance_id, config.ready_timeout_seconds)
            ssh_target = wait_for_ssh(api, instance_id, config.ssh_timeout_seconds)
            remote_runner(
                ssh_target,
                selection,
                config,
                run_name,
                wandb_key=wandb_key,
                github_token=github_token,
            )
        except BaseException as exc:
            failure = exc
        finally:
            if instance_id is not None:
                print(f"Destroying Vast instance {instance_id}.")
                destroyed = destroy_with_retries(
                    api,
                    instance_id,
                    retries=config.destroy_retries,
                )
                if destroyed:
                    state_path.unlink(missing_ok=True)
                    print(f"Destroyed Vast instance {instance_id}.")
                else:
                    message = (
                        f"could not confirm destruction of Vast instance {instance_id}; "
                        f"state retained at {state_path}. Run: vastai destroy instance "
                        f"{instance_id}"
                    )
                    failure = RuntimeError(message)
        if failure is not None:
            raise failure
    return 0


def shell_preview(command: Sequence[str]) -> str:
    """Format a command for diagnostics without interpolating secrets."""

    return shlex.join(command)
