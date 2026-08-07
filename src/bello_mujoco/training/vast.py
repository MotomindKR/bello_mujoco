"""Vast.ai provisioning and remote training orchestration."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
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
from typing import Literal
from urllib.parse import urlparse

import keyring

from bello_mujoco.configuration import ExperimentName, profile_config_relative_path
from bello_mujoco.training.vast_provider import (
    VastApi,
    VastOffer,
    VastSdkClient,
    exception_http_status,
)
from bello_mujoco.training.vast_remote import (
    run_remote_training,
    wait_for_ssh,
)


VAST_KEYRING_SERVICE = "vast"
VAST_KEYRING_USERNAME = "api_key"
WANDB_KEYRING_SERVICE = "wandb"
WANDB_KEYRING_USERNAME = "api_key"
GITHUB_KEYRING_SERVICE = "github"
GITHUB_KEYRING_USERNAME = "bello_mujoco_read_token"
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


@dataclass(frozen=True)
class VastTrainConfig:
    """Resolved configuration for one complete Vast training lifecycle."""

    experiment: ExperimentName = "hardware_locomotion"
    motion_path: Path | None = None
    dataset_path: Path | None = None
    teacher_checkpoint_path: Path | None = None
    allow_git_commit_mismatch: bool = False
    max_price: float | None = None
    branch: str | None = None
    run_name: str | None = None
    skip_confirmation: bool = False
    dry_run: bool = False
    remote_dir: str = "/workspace/bello_mujoco"
    remote_dataset_dir: str = "/workspace/datasets/twist"
    remote_teacher_checkpoint_dir: str = "/workspace/checkpoints/twist_teacher"
    lifecycle_state_dir: Path = Path(".vast")
    ready_timeout_seconds: int = 900
    ssh_timeout_seconds: int = 900
    transfer_timeout_seconds: int = 3_600
    transfer_retries: int = 3
    job_timeout_seconds: int = 43_200
    progress_timeout_seconds: int = 1_800
    reconcile_retries: int = 3
    destroy_retries: int = 3
    offer: VastOfferConfig = field(default_factory=VastOfferConfig)


@dataclass(frozen=True)
class GitSelection:
    branch: str
    commit: str
    config_path: PurePosixPath
    repo_url: str
    motion_path: PurePosixPath | None = None
    dataset_fingerprint: str | None = None
    dataset_files: tuple[Path, ...] = ()
    teacher_checkpoint_path: Path | None = None


@dataclass(frozen=True)
class VastLifecycleState:
    """Durable state for reconciling a possibly billed Vast instance."""

    phase: Literal["creating", "active"]
    label: str
    instance_id: int | None


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
    if config.experiment not in {
        "smoke",
        "locomotion",
        "hardware_locomotion",
        "motion",
        "twist_teacher",
        "twist_student",
    }:
        raise ValueError("--experiment is invalid")
    if config.experiment == "motion":
        if config.motion_path is None:
            raise ValueError("--motion-path is required for the motion experiment")
        _repository_motion_path(config.motion_path)
    elif config.motion_path is not None:
        raise ValueError("--motion-path is only valid for the motion experiment")
    if config.experiment in {"twist_teacher", "twist_student"}:
        if config.dataset_path is None:
            raise ValueError("--dataset-path is required for TWIST Vast training")
    elif config.dataset_path is not None:
        raise ValueError("--dataset-path is only valid for TWIST Vast training")
    if config.experiment == "twist_student":
        if config.teacher_checkpoint_path is None:
            raise ValueError(
                "--teacher-checkpoint-path is required for TWIST student Vast training"
            )
    elif config.teacher_checkpoint_path is not None:
        raise ValueError(
            "--teacher-checkpoint-path is only valid for TWIST student Vast training"
        )
    if config.allow_git_commit_mismatch and config.experiment != "twist_student":
        raise ValueError(
            "--allow-git-commit-mismatch is only valid for TWIST student Vast training"
        )
    if (
        config.max_price is None
        or not math.isfinite(config.max_price)
        or config.max_price <= 0
    ):
        raise ValueError("--max-price must be finite and positive")
    if not config.branch:
        raise ValueError("--branch is required")
    if config.ready_timeout_seconds <= 0 or config.ssh_timeout_seconds <= 0:
        raise ValueError("readiness timeouts must be positive")
    if config.transfer_timeout_seconds <= 0 or config.transfer_retries <= 0:
        raise ValueError("transfer timeout and retries must be positive")
    if config.job_timeout_seconds <= 0 or config.progress_timeout_seconds <= 0:
        raise ValueError("job and progress timeouts must be positive")
    if config.progress_timeout_seconds > config.job_timeout_seconds:
        raise ValueError("progress-timeout-seconds must not exceed job-timeout-seconds")
    if config.reconcile_retries <= 0 or config.destroy_retries <= 0:
        raise ValueError("reconcile-retries and destroy-retries must be positive")
    remote_paths = {
        "--remote-dir": PurePosixPath(config.remote_dir),
        "--remote-dataset-dir": PurePosixPath(config.remote_dataset_dir),
        "--remote-teacher-checkpoint-dir": PurePosixPath(
            config.remote_teacher_checkpoint_dir
        ),
    }
    workspace = PurePosixPath("/workspace")
    for label, remote_path in remote_paths.items():
        raw = {
            "--remote-dir": config.remote_dir,
            "--remote-dataset-dir": config.remote_dataset_dir,
            "--remote-teacher-checkpoint-dir": config.remote_teacher_checkpoint_dir,
        }[label]
        if (
            not remote_path.is_absolute()
            or remote_path.anchor != "/"
            or workspace not in remote_path.parents
            or ".." in remote_path.parts
            or str(remote_path) != raw
            or re.fullmatch(r"/[A-Za-z0-9._/-]+", raw) is None
        ):
            raise ValueError(f"{label} must be a normalized strict child of /workspace")
    if len(set(remote_paths.values())) != len(remote_paths):
        raise ValueError("remote repository, dataset, and checkpoint paths must differ")
    state_dir = config.lifecycle_state_dir
    if (
        state_dir.is_absolute()
        or not state_dir.parts
        or state_dir.parts[0] != ".vast"
        or ".." in state_dir.parts
    ):
        raise ValueError(
            "--lifecycle-state-dir must be a relative path inside .vast, got "
            f"{state_dir}"
        )
    offer = config.offer
    if offer.num_gpus <= 0 or offer.min_cpu_ram_gb <= 0 or offer.disk_gb <= 0:
        raise ValueError("GPU count, CPU RAM, and disk size must be positive")
    if not 0.0 < offer.min_reliability <= 1.0:
        raise ValueError("offer.min-reliability must be in (0, 1]")
    if offer.search_limit <= 0:
        raise ValueError("offer.search-limit must be positive")


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


def resolve_git_selection(repo_root: Path, config: VastTrainConfig) -> GitSelection:
    branches = remote_branches(repo_root)
    branch = config.branch
    assert branch is not None
    if branch not in branches:
        raise ValueError(f"branch {branch!r} does not exist on origin")

    config_path = PurePosixPath(
        profile_config_relative_path(config.experiment).as_posix()
    )

    ref = f"refs/remotes/origin/{branch}"
    commit = run_git(repo_root, "rev-parse", f"{ref}^{{commit}}")
    for path in (
        str(config_path),
        ".gitmodules",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    ):
        try:
            run_git(repo_root, "cat-file", "-e", f"{commit}:{path}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{path!r} does not exist on origin/{branch}") from exc
    submodule_path = "deps/bello-stack"
    submodule_entry = run_git(
        repo_root, "ls-tree", commit, "--", submodule_path
    ).split()
    if (
        len(submodule_entry) != 4
        or submodule_entry[:2] != ["160000", "commit"]
        or submodule_entry[3] != submodule_path
    ):
        raise ValueError(f"{submodule_path!r} is not a submodule on origin/{branch}")
    motion_path = None
    if config.motion_path is not None:
        motion_path = _repository_motion_path(config.motion_path)
        try:
            run_git(repo_root, "cat-file", "-e", f"{commit}:{motion_path}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                "motion-tracking Vast runs require a repository-tracked clip; "
                f"{motion_path!s} is absent from origin/{branch}. Ignored local "
                "datasets are never uploaded implicitly."
            ) from exc
    dataset_fingerprint = None
    dataset_files: tuple[Path, ...] = ()
    if config.dataset_path is not None:
        dataset_fingerprint, dataset_files = _twist_dataset_files(config.dataset_path)
    teacher_checkpoint_path = None
    if config.teacher_checkpoint_path is not None:
        teacher_checkpoint_path = _twist_teacher_checkpoint(
            config.teacher_checkpoint_path,
            dataset_fingerprint=dataset_fingerprint,
        )
    repo_url = normalized_repo_url(run_git(repo_root, "remote", "get-url", "origin"))
    return GitSelection(
        branch=branch,
        commit=commit,
        config_path=config_path,
        repo_url=repo_url,
        motion_path=motion_path,
        dataset_fingerprint=dataset_fingerprint,
        dataset_files=dataset_files,
        teacher_checkpoint_path=teacher_checkpoint_path,
    )


def _twist_dataset_files(index_path: Path) -> tuple[str, tuple[Path, ...]]:
    from bello_mujoco.motion.twist_dataset import (
        TWIST_DATASET_SCHEMA_VERSION,
        TWIST_TRAIN_SAMPLING,
    )

    index_path = Path(index_path).expanduser().resolve()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"TWIST dataset index does not exist: {index_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"TWIST dataset index is invalid JSON: {index_path}") from exc
    if index.get("schema_version") != TWIST_DATASET_SCHEMA_VERSION:
        raise ValueError("Vast TWIST training requires the current dataset schema")
    if index.get("train_sampling") != TWIST_TRAIN_SAMPLING:
        raise ValueError("Vast TWIST dataset has an unsupported sampling contract")
    split_files = index.get("split_files")
    if split_files != {"train": "train.npz", "evaluation": "evaluation.npz"}:
        raise ValueError("Vast TWIST dataset has an unsupported split file contract")
    fingerprint = index.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError("Vast TWIST dataset fingerprint is invalid")
    root = index_path.parent
    paths = (
        index_path,
        root / "train.npz",
        root / "evaluation.npz",
        root / "heldout_manifest.json",
        root / "conversion_report.json",
    )
    missing = [str(path) for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError("Vast TWIST dataset files are missing: " + ", ".join(missing))
    return fingerprint, paths


def _twist_teacher_checkpoint(
    checkpoint_path: Path, *, dataset_fingerprint: str | None
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if any(path.is_symlink() for path in checkpoint_path.rglob("*")):
        raise ValueError("TWIST teacher checkpoint must not contain symbolic links")
    metadata_path = checkpoint_path / "metadata.json"
    try:
        envelope = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"TWIST teacher checkpoint metadata does not exist: {metadata_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("TWIST teacher checkpoint metadata is invalid JSON") from exc
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("task") != "twist_teacher":
        raise ValueError("Vast TWIST student requires a typed teacher checkpoint")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict) or (
        dataset.get("dataset_fingerprint") != dataset_fingerprint
    ):
        raise ValueError("TWIST teacher checkpoint dataset fingerprint differs")
    for name in ("state", "continuation"):
        if not (checkpoint_path / name).is_dir():
            raise ValueError(f"TWIST teacher checkpoint is missing {name}/")
    return checkpoint_path


def _repository_motion_path(value: Path) -> PurePosixPath:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("motion clip must be a repository-relative path")
    if path.suffix != ".npz" or path.name == ".npz":
        raise ValueError("motion_path must name an .npz clip")
    if path.parts[:2] == ("motions", "local"):
        raise ValueError(
            "motions/local is ignored and is not uploaded to Vast; choose a "
            "repository-tracked motion clip"
        )
    return path


def _task_name(config: VastTrainConfig) -> str:
    if config.experiment == "motion":
        return "motion-tracking"
    if config.experiment in {"twist_teacher", "twist_student"}:
        return config.experiment.replace("_", "-")
    return "locomotion"


def normalized_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    if parsed.scheme != "https":
        raise ValueError("origin must be an HTTPS Git remote for remote cloning")
    if parsed.username or parsed.password:
        raise ValueError(
            "origin URL must not embed credentials; use the GitHub keyring"
        )
    return repo_url


def offer_query(config: VastTrainConfig) -> str:
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
    return " ".join(parts)


def find_offers(api: VastApi, config: VastTrainConfig) -> list[VastOffer]:
    raw_offers = api.search_offers(
        offer_query(config),
        order="dlperf_per_dphtotal-",
        limit=config.offer.search_limit,
        storage=float(config.offer.disk_gb),
    )
    offers = [VastOffer.from_mapping(item) for item in raw_offers]
    offers = [offer for offer in offers if offer_matches(offer, config)]
    offers.sort(key=lambda offer: (-offer.dlperf_per_dphtotal, offer.dph_total))
    if offers:
        return offers
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


def print_offers(offers: Sequence[VastOffer], *, limit: int = 5) -> None:
    print("Matching Vast offers (best value first):")
    print("  offer       $/hr   reliability  location             CPU RAM  GPU RAM")
    for offer in offers[:limit]:
        print(
            f"  {offer.id:<10d}  {offer.dph_total:>5.3f}  "
            f"{offer.reliability:>10.3f}  {offer.geolocation[:20]:<20} "
            f"{offer.cpu_ram_mb / 1024:>6.1f}G  {offer.gpu_ram_mb / 1024:>6.1f}G"
        )


def confirm_rental(offer: VastOffer) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "skip_confirmation=true is required when stdin is not interactive"
        )
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
        ),
    )


def write_active_state(state_path: Path, instance_id: int, label: str) -> None:
    _write_lifecycle_state(
        state_path,
        VastLifecycleState(
            phase="active",
            label=label,
            instance_id=instance_id,
        ),
    )


def _write_lifecycle_state(state_path: Path, state: VastLifecycleState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "phase": state.phase,
                "instance_id": state.instance_id,
                "label": state.label,
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
    expected_fields = {"phase", "instance_id", "label"}
    if set(value) != expected_fields:
        raise ValueError(
            f"invalid Vast lifecycle state fields at {state_path}: "
            f"expected {sorted(expected_fields)}, got {sorted(value)}"
        )

    phase = value["phase"]
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
    return VastLifecycleState(
        phase=phase,
        label=label,
        instance_id=instance_id,
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
        return _is_explicit_not_found(exc)
    if info is None:
        return True
    return info.get("actual_status") == "destroyed"


def _is_explicit_not_found(exc: BaseException) -> bool:
    return exception_http_status(exc) == 404


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
    on_before_create: Callable[[], None],
    on_offer_unavailable: Callable[[], None],
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Rent the first still-available offer without exposing SDK error URLs."""

    for offer in offers:
        if not config.skip_confirmation:
            confirm_rental(offer)
        on_before_create()
        try:
            return api.create_instance(
                offer.id,
                image=config.offer.image,
                disk=float(config.offer.disk_gb),
                label=label,
                runtype="ssh_direc",
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
            on_offer_unavailable()
            print(
                f"Vast offer {offer.id} could not be rented "
                f"({type(exc).__name__}); trying the next match."
            )
    raise RuntimeError("all matching Vast offers became unavailable during rental")


def _is_explicit_offer_unavailable(exc: BaseException) -> bool:
    return exception_http_status(exc) in {404, 410}


def watchdog_deadline(config: VastTrainConfig, *, now: float | None = None) -> int:
    created_at = time.time() if now is None else now
    return int(created_at) + (
        config.ready_timeout_seconds
        + config.ssh_timeout_seconds
        + config.transfer_timeout_seconds
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
        if info is None:
            raise RuntimeError(f"Vast instance {instance_id} is absent")
        try:
            status = str(info["actual_status"])
        except KeyError as exc:
            raise RuntimeError(
                f"Vast instance {instance_id} response has no actual_status"
            ) from exc
        print(f"Vast instance {instance_id}: {status}")
        if status == "running":
            return
        if status in {"exited", "offline"}:
            raise RuntimeError(f"instance entered non-recoverable state {status!r}")
        sleep(10)
    raise TimeoutError(f"instance {instance_id} did not become ready in time")


class PipelineInterrupted(RuntimeError):
    pass


@contextmanager
def termination_handler():
    handled_signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
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
        offers = find_offers(self.api, self.config)
        print_offers(offers)
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
        f"{selection.config_path} for {_task_name(config)} (run {run_name})."
    )
    if selection.motion_path is not None:
        print(f"Selected repository-tracked motion {selection.motion_path}.")

    if config.dry_run:
        print(f"Offer query: {offer_query(config)}")
        print("Dry run: no credentials were read and no instance was rented.")
        return 0

    if selection.dataset_files and shutil.which("rsync") is None:
        raise RuntimeError("TWIST Vast training requires rsync on the launch machine")

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
    state_dir = repo_root / config.lifecycle_state_dir
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


def safe_exception_message(exc: BaseException) -> str:
    """Redact credentials that third-party HTTP clients embed in URLs."""

    message = str(exc)
    return re.sub(
        r"([?&](?:api_key|token)=)[^&\s'\"]+",
        r"\1<redacted>",
        message,
        flags=re.IGNORECASE,
    )
