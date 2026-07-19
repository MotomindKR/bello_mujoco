"""Vast.ai provisioning and remote training orchestration."""

from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import keyring
import yaml


VAST_KEYRING_SERVICE = "vast"
VAST_KEYRING_USERNAME = "api_key"
WANDB_KEYRING_SERVICE = "wandb"
WANDB_KEYRING_USERNAME = "api_key"
GITHUB_KEYRING_SERVICE = "github"
GITHUB_KEYRING_USERNAME = "bello_mujoco_read_token"
LIFECYCLE_STATE_VERSION = 1
SSH_SUBPROCESS_TIMEOUT_SECONDS = 60.0
WATCHDOG_TEARDOWN_GRACE_SECONDS = 900


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
    task: Literal["locomotion", "motion-tracking"] = "locomotion"
    max_price: float | None = None
    branch: str | None = None
    run_name: str | None = None
    yes: bool = False
    dry_run: bool = False
    remote_dir: str = "/workspace/bello_mujoco"
    ready_timeout_seconds: int = 900
    ssh_timeout_seconds: int = 900
    job_timeout_seconds: int = 43_200
    progress_timeout_seconds: int = 1_800
    reconcile_retries: int = 3
    destroy_retries: int = 3
    offer: VastOfferConfig = field(default_factory=VastOfferConfig)
    hydra_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitSelection:
    branch: str
    commit: str
    config_path: PurePosixPath
    repo_url: str
    motion_path: PurePosixPath | None = None


@dataclass(frozen=True)
class RemoteJobUpdate:
    """One bounded log/status response from a detached remote job."""

    state: str
    offset: int
    output: bytes
    returncode: int | None = None


@dataclass(frozen=True)
class VastLifecycleState:
    """Durable state for reconciling a possibly billed Vast instance."""

    phase: str
    label: str
    instance_id: int | None
    created_at: str


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
            if "offers" in value:
                value = value["offers"]
            elif "instances" in value:
                value = value["instances"]
            elif ("id" in value or "ask_contract_id" in value) and "gpu_name" in value:
                value = [value]
            else:
                raise RuntimeError(f"unexpected Vast offer response: {value!r}")
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            raise RuntimeError(f"unexpected Vast offer response: {value!r}")
        if any(
            not isinstance(item, Mapping)
            or ("id" not in item and "ask_contract_id" not in item)
            for item in value
        ):
            raise RuntimeError(f"unexpected Vast offer response: {value!r}")
        return value

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
            if len(value) != 1:
                raise RuntimeError(f"unexpected Vast instance response: {value!r}")
            value = value[0]
        if not isinstance(value, Mapping) or "id" not in value:
            raise RuntimeError(f"unexpected Vast instance response: {value!r}")
        return value

    def show_instances(self) -> list[Mapping[str, Any]]:
        value = _json_value(self._client.show_instances())
        if isinstance(value, Mapping):
            if "instances" in value:
                value = value["instances"]
            elif "id" in value and (
                "actual_status" in value or "cur_state" in value or "label" in value
            ):
                value = [value]
            else:
                raise RuntimeError(f"unexpected Vast instances response: {value!r}")
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            raise RuntimeError(f"unexpected Vast instances response: {value!r}")
        if any(not isinstance(item, Mapping) or "id" not in item for item in value):
            raise RuntimeError(f"unexpected Vast instances response: {value!r}")
        return value

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
    if config.task not in {"locomotion", "motion-tracking"}:
        raise ValueError("--task must be 'locomotion' or 'motion-tracking'")
    if config.max_price is None or config.max_price <= 0:
        raise ValueError("--max-price must be positive")
    if config.ready_timeout_seconds <= 0 or config.ssh_timeout_seconds <= 0:
        raise ValueError("readiness timeouts must be positive")
    if config.job_timeout_seconds <= 0 or config.progress_timeout_seconds <= 0:
        raise ValueError("job and progress timeouts must be positive")
    if config.progress_timeout_seconds > config.job_timeout_seconds:
        raise ValueError("progress-timeout-seconds must not exceed job-timeout-seconds")
    if config.reconcile_retries <= 0 or config.destroy_retries <= 0:
        raise ValueError("reconcile-retries and destroy-retries must be positive")
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
    motion_path = None
    if config.task == "motion-tracking":
        config_text = run_git(repo_root, "show", f"{commit}:{config_path}")
        motion_path = resolve_vast_motion_path(config_text, config.hydra_overrides)
        try:
            run_git(repo_root, "cat-file", "-e", f"{commit}:{motion_path}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                "motion-tracking Vast runs require a repository-tracked clip; "
                f"{motion_path!s} is absent from origin/{branch}. Ignored local "
                "datasets are never uploaded implicitly."
            ) from exc
    repo_url = normalized_repo_url(run_git(repo_root, "remote", "get-url", "origin"))
    return GitSelection(
        branch=branch,
        commit=commit,
        config_path=config_path,
        repo_url=repo_url,
        motion_path=motion_path,
    )


def resolve_vast_motion_path(
    config_text: str, hydra_overrides: Sequence[str] = ()
) -> PurePosixPath:
    """Resolve the tracked clip selected by a motion recipe and its overrides."""

    try:
        document = yaml.safe_load(config_text) or {}
    except yaml.YAMLError as error:
        raise ValueError("motion training config is not valid YAML") from error
    if not isinstance(document, Mapping):
        raise ValueError("motion training config must contain a YAML mapping")
    motion = document.get("motion", {})
    if not isinstance(motion, Mapping):
        raise ValueError("motion training config motion field must be a mapping")
    directory = motion.get("directory")
    sequence = motion.get("sequence")
    for override in hydra_overrides:
        key, separator, value = override.partition("=")
        key = key.lstrip("+")
        if not separator or key not in {"motion.directory", "motion.sequence"}:
            continue
        try:
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid Hydra override {override!r}") from error
        if key == "motion.directory":
            directory = parsed_value
        else:
            sequence = parsed_value
    if not isinstance(directory, str) or not directory:
        raise ValueError(
            "motion-tracking Vast config must select motion.directory explicitly"
        )
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(
            "motion-tracking Vast config must select motion.sequence explicitly"
        )
    sequence_path = PurePosixPath(sequence)
    if sequence_path.name != sequence or sequence_path.suffix:
        raise ValueError("motion.sequence must be a name without a path or suffix")
    path = PurePosixPath(directory) / f"{sequence}.npz"
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("motion clip must be a repository-relative path")
    if path.parts[:2] == ("motions", "local"):
        raise ValueError(
            "motions/local is ignored and is not uploaded to Vast; choose a "
            "repository-tracked motion clip"
        )
    return path


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


def write_creating_state(state_path: Path, label: str) -> None:
    _write_lifecycle_state(
        state_path,
        VastLifecycleState(
            phase="creating",
            label=label,
            instance_id=None,
            created_at=datetime.now(UTC).isoformat(),
        ),
    )


def write_active_state(state_path: Path, instance_id: int, label: str) -> None:
    _write_lifecycle_state(
        state_path,
        VastLifecycleState(
            phase="active",
            label=label,
            instance_id=instance_id,
            created_at=datetime.now(UTC).isoformat(),
        ),
    )


def _write_lifecycle_state(state_path: Path, state: VastLifecycleState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "version": LIFECYCLE_STATE_VERSION,
                "phase": state.phase,
                "instance_id": state.instance_id,
                "label": state.label,
                "created_at": state.created_at,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=state_path.parent,
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, state_path)
        directory_fd = os.open(state_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_lifecycle_state(state_path: Path) -> VastLifecycleState | None:
    if not state_path.exists():
        return None
    value = json.loads(state_path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid Vast lifecycle state at {state_path}")

    version = value.get("version")
    if version is None:
        # Legacy state contained only a confirmed active ID. Reading this narrow
        # format is intentional because abandoning it could leak a billed instance.
        phase = "active"
    elif type(version) is not int or version != LIFECYCLE_STATE_VERSION:
        raise ValueError(f"unsupported Vast lifecycle state version {version!r}")
    else:
        phase = value.get("phase")
    if phase not in {"creating", "active"}:
        raise ValueError(f"invalid Vast lifecycle phase {phase!r}")

    label = value.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("Vast lifecycle state requires a nonempty label")
    instance_id = value.get("instance_id")
    if phase == "creating":
        if instance_id is not None:
            raise ValueError("creating Vast lifecycle state must not contain an ID")
    elif type(instance_id) is not int or instance_id <= 0:
        raise ValueError("active Vast lifecycle state requires a positive instance ID")
    created_at = value.get("created_at", "")
    if not isinstance(created_at, str):
        raise ValueError("Vast lifecycle state created_at must be a string")
    return VastLifecycleState(
        phase=phase,
        label=label,
        instance_id=instance_id,
        created_at=created_at,
    )


def destroy_with_retries(
    api: VastApi,
    instance_id: int,
    *,
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    for attempt in range(1, retries + 1):
        try:
            api.destroy_instance(instance_id)
        except Exception as exc:
            if _is_explicit_not_found(exc):
                return True
        if _instance_absent_or_destroyed(api, instance_id):
            return True
        if attempt < retries:
            sleep(min(2**attempt, 10))
    return False


def _instance_absent_or_destroyed(api: VastApi, instance_id: int) -> bool:
    try:
        info = api.show_instance(instance_id)
    except Exception as exc:
        if _is_explicit_not_found(exc):
            return True
    else:
        if info.get("cur_state") == "unloaded" or info.get("actual_status") == (
            "destroyed"
        ):
            return True
        if info.get("actual_status") is not None:
            return False

    try:
        instances = api.show_instances()
    except Exception:
        return False
    return all(int(item.get("id", -1)) != instance_id for item in instances)


def _is_explicit_not_found(exc: BaseException) -> bool:
    if _exception_http_status(exc) == 404:
        return True
    message = str(exc).lower()
    return bool(
        re.fullmatch(r"\s*404(?:\s+not found)?\s*", message)
        or re.search(r"\binstance\b.{0,80}\bnot found\b", message)
        or re.search(r"\binstance\b.{0,80}\bdoes not exist\b", message)
        or re.search(r"\bno such instance\b", message)
    )


def _exception_http_status(exc: BaseException) -> int | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        for owner, attribute in (
            (current, "status_code"),
            (current, "http_status"),
            (response, "status_code"),
        ):
            raw_status = getattr(owner, attribute, None)
            if type(raw_status) is int:
                return raw_status
            if isinstance(raw_status, str) and raw_status.isdigit():
                return int(raw_status)
        current = current.__cause__ or current.__context__
    return None


def reconcile_stale_state(
    api: VastApi,
    state_path: Path,
    *,
    reconcile_retries: int,
    destroy_retries: int,
) -> None:
    state = read_lifecycle_state(state_path)
    if state is None:
        return
    if state.phase == "creating":
        instance_id = recover_labeled_instance(
            api,
            state.label,
            retries=reconcile_retries,
        )
        if instance_id is None:
            raise RuntimeError(
                f"pending Vast rental {state.label!r} is still unresolved; state "
                f"retained at {state_path} and another rental is refused"
            )
        write_active_state(state_path, instance_id, state.label)
        print(
            f"Cleaning reconciled pipeline instance {instance_id} "
            f"({state.label}) before launch."
        )
        if not destroy_with_retries(api, instance_id, retries=destroy_retries):
            raise RuntimeError(
                f"could not destroy reconciled instance {instance_id}; "
                "refusing another rental"
            )
        state_path.unlink(missing_ok=True)
        return

    assert state.instance_id is not None
    instance_id = state.instance_id

    matching_instances = [
        item for item in api.show_instances() if int(item.get("id", -1)) == instance_id
    ]
    if not matching_instances:
        state_path.unlink(missing_ok=True)
        return
    actual_label = matching_instances[0].get("label")
    if actual_label != state.label:
        raise RuntimeError(
            f"stale state label {state.label!r} does not match instance {instance_id} "
            f"label {actual_label!r}; refusing destructive cleanup"
        )
    print(
        f"Cleaning stale pipeline instance {instance_id} ({state.label}) before launch."
    )
    if not destroy_with_retries(api, instance_id, retries=destroy_retries):
        raise RuntimeError(
            f"could not destroy stale instance {instance_id}; refusing another rental"
        )
    state_path.unlink(missing_ok=True)


def recover_labeled_instance(
    api: VastApi,
    label: str,
    *,
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    successful_listing = False
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            matches = [
                item for item in api.show_instances() if item.get("label") == label
            ]
            successful_listing = True
        except Exception as exc:
            last_error = exc
        else:
            if len(matches) > 1:
                ids = [item.get("id") for item in matches]
                raise RuntimeError(
                    f"multiple Vast instances use reconciliation label {label!r}: {ids}; "
                    "refusing another rental"
                )
            if matches:
                value = matches[0].get("id")
                if value is None:
                    raise RuntimeError(
                        f"Vast instance with label {label!r} has no instance id"
                    )
                return int(value)
        if attempt < retries:
            sleep(min(2**attempt, 10))
    if not successful_listing:
        raise RuntimeError(
            f"could not reconcile ambiguous Vast rental for label {label!r}"
        ) from last_error
    return None


def create_available_instance(
    api: VastApi,
    offers: Sequence[VastOffer],
    config: VastTrainConfig,
    label: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_before_create: Callable[[], None] | None = None,
    on_offer_unavailable: Callable[[], None] | None = None,
) -> int:
    """Rent the first still-available offer without exposing SDK error URLs."""

    for offer in offers:
        if not config.yes:
            confirm_rental(offer)
        if on_before_create is not None:
            on_before_create()
        try:
            return api.create_instance(
                offer.id,
                image=config.offer.image,
                disk=float(config.offer.disk_gb),
                label=label,
                runtype="ssh_direc ssh_proxy",
                cancel_unavail=True,
                onstart_cmd=watchdog_onstart_script(config),
            )
        except Exception as exc:
            instance_id = recover_labeled_instance(
                api,
                label,
                retries=config.reconcile_retries,
                sleep=sleep,
            )
            if instance_id is not None:
                return instance_id
            if not _is_explicit_offer_unavailable(exc):
                raise RuntimeError(
                    f"Vast rental result for offer {offer.id} is ambiguous; "
                    f"no instance with label {label!r} was visible after "
                    "reconciliation, so another offer will not be rented"
                ) from exc
            if on_offer_unavailable is not None:
                on_offer_unavailable()
            print(
                f"Vast offer {offer.id} could not be rented "
                f"({type(exc).__name__}); trying the next match."
            )
    raise RuntimeError("all matching Vast offers became unavailable during rental")


def _is_explicit_offer_unavailable(exc: BaseException) -> bool:
    if _exception_http_status(exc) in {404, 410}:
        return True
    message = str(exc).lower()
    return bool(
        re.fullmatch(r"\s*(?:404|410)(?:\s+(?:gone|not found))?\s*", message)
        or re.search(
            r"\boffer\b.{0,80}\b(?:disappeared|unavailable|not rentable)\b",
            message,
        )
    )


def watchdog_deadline(config: VastTrainConfig, *, now: float | None = None) -> int:
    created_at = time.time() if now is None else now
    return int(created_at) + (
        config.ready_timeout_seconds
        + config.ssh_timeout_seconds
        + config.job_timeout_seconds
        + WATCHDOG_TEARDOWN_GRACE_SECONDS
    )


def watchdog_onstart_script(config: VastTrainConfig) -> str:
    program = _watchdog_program(watchdog_deadline(config))
    return (
        f"set +x\nnohup python3 -c {shlex.quote(program)} </dev/null >/dev/null 2>&1 &"
    )


def _watchdog_program(deadline: int) -> str:
    return f"""import os
import time
import urllib.error
import urllib.request

deadline = {deadline}
time.sleep(max(0.0, deadline - time.time()))
instance_id = os.environ.get("CONTAINER_ID", "")
api_key = os.environ.get("CONTAINER_API_KEY", "")
if not instance_id.isdigit() or not api_key:
    raise SystemExit(0)
url = f"https://console.vast.ai/api/v0/instances/{{instance_id}}/"
for attempt in range(6):
    request = urllib.request.Request(
        url,
        method="DELETE",
        headers={{"Authorization": f"Bearer {{api_key}}"}},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SystemExit(0)
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(min(30 * (2 ** attempt), 300))
"""


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


def direct_ssh_target(info: Mapping[str, Any]) -> tuple[str, str, int] | None:
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
        host_port = mapping.get("HostPort")
        try:
            port = int(host_port)
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


def wait_for_ssh(
    api: VastApi,
    instance_id: int,
    timeout_seconds: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
    executor: SshExecutor | None = None,
) -> tuple[str, str, int]:
    executor = executor or SshExecutor()
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        targets: list[tuple[str, str, int]] = []
        try:
            direct_target = direct_ssh_target(api.show_instance(instance_id))
            if direct_target is not None:
                targets.append(direct_target)
            proxy_target = parse_ssh_url(api.ssh_url(instance_id))
            if proxy_target not in targets:
                targets.append(proxy_target)
        except (OSError, ValueError) as exc:
            last_error = exc
        for target in targets:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                executor.probe(
                    target,
                    timeout_seconds=min(SSH_SUBPROCESS_TIMEOUT_SECONDS, remaining),
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
    raise TimeoutError(f"SSH did not become ready: {last_error}")


REMOTE_TRAIN_SCRIPT = r"""set -Eeuo pipefail

branch="$1"
expected_commit="$2"
repo_url="$3"
remote_dir="$4"
config_path="$5"
run_name="$6"
task="$7"

case "$task" in
  locomotion) train_command="bello-train-ppo" ;;
  motion-tracking) train_command="bello-train-motion-ppo" ;;
  *) echo "unsupported Vast training task: $task" >&2; exit 2 ;;
esac

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
export JAX_DEFAULT_MATMUL_PRECISION=highest

uv python install 3.12
uv sync --frozen
uv run python -c 'import jax; devices=jax.devices(); print({"backend": jax.default_backend(), "devices": [str(x) for x in devices]}); assert jax.default_backend() == "gpu"'

config_dir="$remote_dir/$(dirname "$config_path")"
config_name="$(basename "$config_path")"
config_name="${config_name%.*}"

mapfile -t user_overrides < <(python3 -c 'import json, os, sys; sys.stdout.write("\n".join(json.loads(os.environ["BELLO_HYDRA_OVERRIDES"])))')

uv run "$train_command" \
  --config-dir "$config_dir" \
  --config-name "$config_name" \
  "${user_overrides[@]}" \
  dry_run=true \
  config.num_envs=1 \
  config.num_steps=1 \
  config.num_minibatches=1 \
  wandb_config.enabled=false

entity_override=()
if [ -n "${WANDB_ENTITY:-}" ]; then
  entity_override+=("wandb_config.entity=$WANDB_ENTITY")
fi
uv run "$train_command" \
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
    executor = executor or SshExecutor()
    project = os.environ.get("WANDB_PROJECT", "bello-mujoco")
    entity = os.environ.get("WANDB_ENTITY", "")
    remote_env = {
        "WANDB_API_KEY": wandb_key,
        "WANDB_PROJECT": project,
        "WANDB_ENTITY": entity,
        "GITHUB_TOKEN": github_token,
        "BELLO_HYDRA_OVERRIDES": json.dumps(config.hydra_overrides),
    }
    job_dir = f"/workspace/.bello-vast-jobs/{run_name}"
    start_command = _remote_start_command(
        job_dir,
        selection,
        config,
        run_name,
    )
    payload = json.dumps({"environment": remote_env, "script": REMOTE_TRAIN_SCRIPT})
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
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            break
        if result.returncode != 255:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
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
                _remote_status_command(job_dir, offset),
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
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        if disconnected_at is not None:
            print("SSH reconnected; resuming remote training output.")
            disconnected_at = None
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        update = _parse_remote_job_update(result.stdout)
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


def _remote_start_command(
    job_dir: str,
    selection: GitSelection,
    config: VastTrainConfig,
    run_name: str,
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
    except (OSError, ValueError):
        pass
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
            str(selection.config_path),
            run_name,
            config.task,
        ]
    )


def _remote_status_command(job_dir: str, offset: int) -> str:
    status_reader = r"""import base64, json, os, pathlib, sys
job_dir = pathlib.Path(sys.argv[1])
offset = int(sys.argv[2])
log_path = job_dir / "output.log"
status_path = job_dir / "status"
pid_path = job_dir / "pid"
size = log_path.stat().st_size if log_path.exists() else 0
if offset > size:
    offset = 0
with log_path.open("rb") if log_path.exists() else open(os.devnull, "rb") as log:
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


def _parse_remote_job_update(value: str) -> RemoteJobUpdate:
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


class PipelineInterrupted(RuntimeError):
    pass


@contextmanager
def termination_handler():
    handled_signals = tuple(
        candidate
        for candidate in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
        if candidate is not None
    )
    previous = {candidate: signal.getsignal(candidate) for candidate in handled_signals}

    def handle_termination(signum: int, frame: FrameType | None) -> None:
        del frame
        raise PipelineInterrupted(f"received {signal.Signals(signum).name}")

    for candidate in handled_signals:
        signal.signal(candidate, handle_termination)
    try:
        yield
    finally:
        for candidate, handler in previous.items():
            signal.signal(candidate, handler)


@dataclass(frozen=True)
class VastLifecycle:
    """Own one recover-rent-run-destroy lifecycle under the pipeline lock."""

    api: VastApi
    config: VastTrainConfig
    state_path: Path
    label: str

    def run(
        self,
        selection: GitSelection,
        run_name: str,
        remote_runner: Callable[..., None],
        *,
        wandb_key: str,
        github_token: str,
    ) -> None:
        reconcile_stale_state(
            self.api,
            self.state_path,
            reconcile_retries=self.config.reconcile_retries,
            destroy_retries=self.config.destroy_retries,
        )
        tier, offers = find_offers(self.api, self.config)
        print_offers(tier, offers)
        instance_id: int | None = None
        failure: BaseException | None = None
        try:
            instance_id = create_available_instance(
                self.api,
                offers,
                self.config,
                self.label,
                on_before_create=lambda: write_creating_state(
                    self.state_path, self.label
                ),
                on_offer_unavailable=lambda: self.state_path.unlink(missing_ok=True),
            )
            write_active_state(self.state_path, instance_id, self.label)
            print(f"Created Vast instance {instance_id}; teardown is armed.")
            wait_for_instance(
                self.api,
                instance_id,
                self.config.ready_timeout_seconds,
            )
            ssh_target = wait_for_ssh(
                self.api,
                instance_id,
                self.config.ssh_timeout_seconds,
            )
            remote_runner(
                ssh_target,
                selection,
                self.config,
                run_name,
                wandb_key=wandb_key,
                github_token=github_token,
            )
        except BaseException as exc:
            failure = exc
        finally:
            if instance_id is not None:
                teardown_failure = self._destroy(instance_id)
                if teardown_failure is not None:
                    failure = teardown_failure
        if failure is not None:
            raise failure

    def _destroy(self, instance_id: int) -> RuntimeError | None:
        print(f"Destroying Vast instance {instance_id}.")
        destroyed = destroy_with_retries(
            self.api,
            instance_id,
            retries=self.config.destroy_retries,
        )
        if destroyed:
            self.state_path.unlink(missing_ok=True)
            print(f"Destroyed Vast instance {instance_id}.")
            return None
        return RuntimeError(
            f"could not confirm destruction of Vast instance {instance_id}; "
            f"state retained at {self.state_path}. Run: vastai destroy instance "
            f"{instance_id}"
        )


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
        f"{selection.config_path} for {config.task} (run {run_name})."
    )
    if selection.motion_path is not None:
        print(f"Selected repository-tracked motion {selection.motion_path}.")

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
        VastLifecycle(api, config, state_path, label).run(
            selection,
            run_name,
            remote_runner,
            wandb_key=wandb_key,
            github_token=github_token,
        )
    return 0


def shell_preview(command: Sequence[str]) -> str:
    """Format a command for diagnostics without interpolating secrets."""

    return shlex.join(command)


def safe_exception_message(exc: BaseException) -> str:
    """Redact credentials that third-party HTTP clients embed in URLs."""

    message = str(exc)
    return re.sub(
        r"([?&](?:api_key|token)=)[^&\s'\"]+",
        r"\1<redacted>",
        message,
        flags=re.IGNORECASE,
    )
