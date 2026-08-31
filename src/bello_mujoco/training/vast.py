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
    measure_upload_mib_per_second,
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
    min_cpu_ram_gb: int = 128
    disk_gb: int = 80
    min_reliability: float = 0.98
    min_cuda: float = 13.0
    image: str = "vastai/base-image:cuda-13.0.3-auto"
    search_limit: int = 50


@dataclass(frozen=True)
class VastTrainConfig:
    """Resolved configuration for one complete Vast training lifecycle."""

    experiment: ExperimentName = "hardware_locomotion"
    motion_path: Path | None = None
    dataset_path: Path | None = None
    teacher_checkpoint_path: Path | None = None
    sonic_artifact_path: Path | None = None
    resume_path: Path | None = None
    warm_start_checkpoint_path: Path | None = None
    allow_git_commit_mismatch: bool = False
    total_timesteps: int | None = None
    num_envs: int | None = None
    checkpoint_every_updates: int | None = None
    learning_rate: float | None = None
    end_effector_payload_mass_kg: float | None = None
    max_price: float | None = None
    branch: str | None = None
    run_name: str | None = None
    skip_confirmation: bool = False
    dry_run: bool = False
    remote_dir: str = "/workspace/bello_mujoco"
    remote_dataset_dir: str = "/workspace/datasets/twist"
    remote_sonic_dataset_dir: str = "/workspace/datasets/sonic"
    remote_teacher_checkpoint_dir: str = "/workspace/checkpoints/twist_teacher"
    remote_sonic_artifact_dir: str = "/workspace/artifacts/sonic/default_release"
    remote_sonic_resume_dir: str = "/workspace/checkpoints/sonic_resume"
    remote_sonic_warm_start_dir: str = "/workspace/checkpoints/sonic_warm_start"
    local_result_dir: Path = Path("outputs/vast")
    remote_warm_start_checkpoint_dir: str = "/workspace/checkpoints/twist2_warm_start"
    lifecycle_state_dir: Path = Path(".vast")
    ready_timeout_seconds: int = 900
    ssh_timeout_seconds: int = 900
    transfer_timeout_seconds: int = 3_600
    transfer_retries: int = 3
    preferred_countries: tuple[str, ...] = ("KR", "JP", "TW", "HK", "SG")
    min_upload_mib_per_second: float = 5.0
    route_probe_mib: int = 256
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
    dataset_root: Path | None = None
    teacher_checkpoint_path: Path | None = None
    sonic_artifact_path: Path | None = None
    resume_checkpoint_path: Path | None = None
    warm_start_checkpoint_path: Path | None = None


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
        "twist2",
        "sonic",
    }:
        raise ValueError("--experiment is invalid")
    if config.experiment == "motion":
        if config.motion_path is None:
            raise ValueError("--motion-path is required for the motion experiment")
        _repository_motion_path(config.motion_path)
        if config.end_effector_payload_mass_kg is not None and (
            not math.isfinite(config.end_effector_payload_mass_kg)
            or config.end_effector_payload_mass_kg < 0.0
        ):
            raise ValueError(
                "--end-effector-payload-mass-kg must be finite and nonnegative"
            )
    elif (
        config.motion_path is not None
        or config.end_effector_payload_mass_kg is not None
    ):
        raise ValueError(
            "--motion-path and --end-effector-payload-mass-kg are only valid "
            "for the motion experiment"
        )
    dataset_experiments = {"twist_teacher", "twist_student", "twist2", "sonic"}
    if config.experiment in dataset_experiments:
        if config.dataset_path is None:
            raise ValueError("--dataset-path is required for dataset-backed training")
    elif config.dataset_path is not None:
        raise ValueError(
            "--dataset-path is only valid for TWIST, TWIST2, or SONIC Vast training"
        )
    if config.experiment == "twist_student":
        if config.teacher_checkpoint_path is None:
            raise ValueError(
                "--teacher-checkpoint-path is required for TWIST student Vast training"
            )
    elif config.teacher_checkpoint_path is not None:
        raise ValueError(
            "--teacher-checkpoint-path is only valid for TWIST student Vast training"
        )
    if config.experiment == "sonic":
        if config.sonic_artifact_path is None:
            raise ValueError(
                "--sonic-artifact-path is required for SONIC Vast training"
            )
        if (
            config.resume_path is not None
            and config.warm_start_checkpoint_path is not None
        ):
            raise ValueError(
                "--resume-path and --warm-start-checkpoint-path are mutually exclusive"
            )
        if config.resume_path is not None:
            missing = [
                name
                for name in (
                    "total_timesteps",
                    "num_envs",
                )
                if getattr(config, name) is None
            ]
            if missing:
                raise ValueError(
                    "resumed SONIC Vast training requires explicit "
                    + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
                )
        elif (
            config.allow_git_commit_mismatch
            and config.warm_start_checkpoint_path is None
        ):
            raise ValueError(
                "--allow-git-commit-mismatch requires --resume-path or "
                "--warm-start-checkpoint-path for SONIC Vast training"
            )
    else:
        if config.sonic_artifact_path is not None:
            raise ValueError(
                "--sonic-artifact-path is only valid for SONIC Vast training"
            )
        if config.resume_path is not None:
            raise ValueError("--resume-path is only valid for SONIC Vast training")
    for name in (
        "total_timesteps",
        "num_envs",
        "checkpoint_every_updates",
    ):
        value = getattr(config, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if (
        config.experiment not in {"twist2", "sonic"}
        and config.warm_start_checkpoint_path is not None
    ):
        raise ValueError(
            "--warm-start-checkpoint-path is only valid for TWIST2 or SONIC Vast "
            "training"
        )
    if config.allow_git_commit_mismatch and not (
        config.experiment == "twist_student"
        or (
            config.experiment == "twist2"
            and config.warm_start_checkpoint_path is not None
        )
        or (config.experiment == "sonic" and config.resume_path is not None)
        or (
            config.experiment == "sonic"
            and config.warm_start_checkpoint_path is not None
        )
    ):
        raise ValueError(
            "--allow-git-commit-mismatch requires a TWIST student teacher or "
            "TWIST2/SONIC continuation checkpoint"
        )
    if config.learning_rate is not None and (
        not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0
    ):
        raise ValueError("--learning-rate must be finite and positive")
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
    if (
        not math.isfinite(config.min_upload_mib_per_second)
        or config.min_upload_mib_per_second <= 0
        or config.route_probe_mib <= 0
    ):
        raise ValueError("route probe size and minimum upload speed must be positive")
    if not config.preferred_countries or any(
        re.fullmatch(r"[A-Z]{2}", country) is None
        for country in config.preferred_countries
    ):
        raise ValueError("preferred countries must be non-empty ISO alpha-2 codes")
    if config.job_timeout_seconds <= 0 or config.progress_timeout_seconds <= 0:
        raise ValueError("job and progress timeouts must be positive")
    if config.progress_timeout_seconds > config.job_timeout_seconds:
        raise ValueError("progress-timeout-seconds must not exceed job-timeout-seconds")
    if config.reconcile_retries <= 0 or config.destroy_retries <= 0:
        raise ValueError("reconcile-retries and destroy-retries must be positive")
    remote_paths = {
        "--remote-dir": PurePosixPath(config.remote_dir),
        "--remote-dataset-dir": PurePosixPath(config.remote_dataset_dir),
        "--remote-sonic-dataset-dir": PurePosixPath(config.remote_sonic_dataset_dir),
        "--remote-teacher-checkpoint-dir": PurePosixPath(
            config.remote_teacher_checkpoint_dir
        ),
        "--remote-sonic-artifact-dir": PurePosixPath(config.remote_sonic_artifact_dir),
        "--remote-sonic-resume-dir": PurePosixPath(config.remote_sonic_resume_dir),
        "--remote-sonic-warm-start-dir": PurePosixPath(
            config.remote_sonic_warm_start_dir
        ),
        "--remote-warm-start-checkpoint-dir": PurePosixPath(
            config.remote_warm_start_checkpoint_dir
        ),
    }
    workspace = PurePosixPath("/workspace")
    for label, remote_path in remote_paths.items():
        raw = {
            "--remote-dir": config.remote_dir,
            "--remote-dataset-dir": config.remote_dataset_dir,
            "--remote-sonic-dataset-dir": config.remote_sonic_dataset_dir,
            "--remote-teacher-checkpoint-dir": config.remote_teacher_checkpoint_dir,
            "--remote-sonic-artifact-dir": config.remote_sonic_artifact_dir,
            "--remote-sonic-resume-dir": config.remote_sonic_resume_dir,
            "--remote-sonic-warm-start-dir": config.remote_sonic_warm_start_dir,
            "--remote-warm-start-checkpoint-dir": (
                config.remote_warm_start_checkpoint_dir
            ),
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
        raise ValueError(
            "remote repository, dataset, checkpoint, and artifact paths must differ"
        )
    local_result_dir = config.local_result_dir
    if (
        local_result_dir.is_absolute()
        or not local_result_dir.parts
        or local_result_dir.parts[0] != "outputs"
        or ".." in local_result_dir.parts
    ):
        raise ValueError("--local-result-dir must be a relative path inside outputs")
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


def fetch_origin(
    repo_root: Path,
    *,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            run_git(repo_root, "fetch", "--prune", "origin")
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < retries:
                print(f"git fetch origin attempt {attempt} failed; retrying.")
                sleep(float(2**attempt))
    assert last_error is not None
    detail = (last_error.stderr or "").strip() or str(last_error)
    raise RuntimeError(
        f"git fetch origin failed after {retries} attempts: "
        f"{safe_exception_message(detail)}"
    ) from last_error


def remote_branches(repo_root: Path) -> list[str]:
    fetch_origin(repo_root)
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
    dataset_root = None
    if config.dataset_path is not None:
        if config.experiment == "sonic":
            dataset_fingerprint, dataset_files, dataset_root = _sonic_dataset_files(
                config.dataset_path
            )
        else:
            dataset_fingerprint, dataset_files = _twist_dataset_files(
                config.dataset_path
            )
    teacher_checkpoint_path = None
    if config.teacher_checkpoint_path is not None:
        teacher_checkpoint_path = _twist_teacher_checkpoint(
            config.teacher_checkpoint_path,
            dataset_fingerprint=dataset_fingerprint,
        )
    sonic_artifact_path = None
    if config.sonic_artifact_path is not None:
        sonic_artifact_path = _sonic_release_artifact(config.sonic_artifact_path)
    resume_checkpoint_path = None
    if config.resume_path is not None:
        assert sonic_artifact_path is not None
        resume_checkpoint_path = _sonic_resume_checkpoint(
            config.resume_path,
            dataset_fingerprint=dataset_fingerprint,
            release_artifact_path=sonic_artifact_path,
            selected_commit=commit,
            allow_git_commit_mismatch=config.allow_git_commit_mismatch,
            total_timesteps=config.total_timesteps,
            num_envs=config.num_envs,
        )
    warm_start_checkpoint_path = None
    if config.warm_start_checkpoint_path is not None:
        assert config.dataset_path is not None
        if config.experiment == "sonic":
            assert sonic_artifact_path is not None
            warm_start_checkpoint_path = _sonic_warm_start_checkpoint(
                config.warm_start_checkpoint_path,
                dataset_fingerprint=dataset_fingerprint,
                release_artifact_path=sonic_artifact_path,
                selected_commit=commit,
                allow_git_commit_mismatch=config.allow_git_commit_mismatch,
            )
        else:
            warm_start_checkpoint_path = _twist2_warm_start_checkpoint(
                config.warm_start_checkpoint_path,
                dataset_path=config.dataset_path,
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
        dataset_root=dataset_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        sonic_artifact_path=sonic_artifact_path,
        resume_checkpoint_path=resume_checkpoint_path,
        warm_start_checkpoint_path=warm_start_checkpoint_path,
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


def _sonic_dataset_files(
    index_path: Path,
) -> tuple[str, tuple[Path, ...], Path]:
    from bello_mujoco.motion.sonic_dataset import load_sonic_dataset_index

    index_path = Path(index_path).expanduser().resolve()
    if index_path.name != "index.json":
        raise ValueError("Vast SONIC dataset path must name index.json")
    try:
        index = load_sonic_dataset_index(index_path)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Vast SONIC dataset is invalid: {exc}") from exc
    root = index_path.parent
    relative_paths = {
        PurePosixPath("index.json"),
        PurePosixPath("heldout_manifest.json"),
        PurePosixPath("conversion_report.json"),
    }
    relative_paths.update(
        PurePosixPath(index["split_files"][split]["file"])
        for split in ("train", "evaluation")
    )
    paths = tuple(root / Path(relative) for relative in sorted(relative_paths))
    missing = [str(path) for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError("Vast SONIC dataset files are missing: " + ", ".join(missing))
    symlinks = [str(path) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(
            "Vast SONIC dataset must not contain symbolic links: " + ", ".join(symlinks)
        )
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected = {relative.as_posix() for relative in relative_paths}
    if actual != expected:
        raise ValueError(
            "Vast SONIC dataset tree differs from its canonical files: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return index["dataset_fingerprint"], paths, root


def _sonic_release_artifact(path: Path) -> Path:
    from bello_mujoco.sonic_policy import load_sonic_release_artifact

    root = Path(path).expanduser().resolve()
    expected = {"manifest.json", "weights.npz", "golden.npz"}
    if not root.is_dir() or any(
        candidate.is_symlink() for candidate in root.rglob("*")
    ):
        raise ValueError(
            "Vast SONIC release artifact must be a directory without symbolic links"
        )
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
    }
    if actual != expected:
        raise ValueError(
            "Vast SONIC release artifact files differ from the pinned export"
        )
    try:
        load_sonic_release_artifact(root)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(f"Vast SONIC release artifact is invalid: {exc}") from exc
    return root


def _sonic_resume_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_fingerprint: str | None,
    release_artifact_path: Path,
    selected_commit: str,
    allow_git_commit_mismatch: bool,
    total_timesteps: int | None,
    num_envs: int | None,
) -> Path:
    from bello_mujoco.sonic_policy import load_sonic_release_artifact
    from bello_mujoco.training.checkpoint import (
        SonicCheckpointMetadata,
        load_checkpoint_metadata,
        load_checkpoint_update,
    )

    unresolved = Path(checkpoint_path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("SONIC resume checkpoint must not be a symbolic link")
    root = unresolved.resolve()
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(
            "SONIC resume checkpoint must be a directory without symbolic links"
        )
    for name in ("metadata.json", "state", "continuation"):
        candidate = root / name
        expected = (
            candidate.is_file() if name == "metadata.json" else candidate.is_dir()
        )
        if not expected:
            raise ValueError(f"SONIC resume checkpoint is missing {name}")
    try:
        metadata = load_checkpoint_metadata(
            root,
            allow_git_commit_mismatch=allow_git_commit_mismatch,
        )
        saved_update = load_checkpoint_update(
            root,
            allow_git_commit_mismatch=allow_git_commit_mismatch,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Vast SONIC resume checkpoint is invalid: {exc}") from exc
    if not isinstance(metadata, SonicCheckpointMetadata):
        raise ValueError("Vast SONIC resume requires a typed SONIC checkpoint")
    if metadata.dataset.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("SONIC resume checkpoint dataset fingerprint differs")
    if metadata.source_revision != selected_commit and not allow_git_commit_mismatch:
        raise ValueError("SONIC resume checkpoint source revision differs")

    release = load_sonic_release_artifact(release_artifact_path)
    if metadata.release_checkpoint_sha256.removeprefix(
        "sha256:"
    ) != release.checkpoint_sha256.removeprefix(
        "sha256:"
    ) or metadata.release_artifact_sha256.removeprefix(
        "sha256:"
    ) != release.artifact_sha256.removeprefix("sha256:"):
        raise ValueError("SONIC resume checkpoint release artifact differs")
    if num_envs != metadata.ppo.num_envs:
        raise ValueError(
            "SONIC resume num environments differs from the checkpoint: "
            f"{num_envs} != {metadata.ppo.num_envs}"
        )
    assert total_timesteps is not None
    requested_updates = total_timesteps // (
        metadata.ppo.num_envs * metadata.ppo.num_steps
    )
    if requested_updates < saved_update:
        raise ValueError(
            "SONIC resume budget ends before the saved checkpoint update: "
            f"{requested_updates} < {saved_update}"
        )
    return root


def _sonic_warm_start_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_fingerprint: str | None,
    release_artifact_path: Path,
    selected_commit: str,
    allow_git_commit_mismatch: bool,
) -> Path:
    from bello_mujoco.sonic_policy import load_sonic_release_artifact
    from bello_mujoco.training.checkpoint import (
        SonicCheckpointMetadata,
        load_checkpoint_metadata,
    )

    unresolved = Path(checkpoint_path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("SONIC warm-start checkpoint must not be a symbolic link")
    root = unresolved.resolve()
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(
            "SONIC warm-start checkpoint must be a directory without symbolic links"
        )
    for name in ("metadata.json", "state", "continuation"):
        candidate = root / name
        expected = (
            candidate.is_file() if name == "metadata.json" else candidate.is_dir()
        )
        if not expected:
            raise ValueError(f"SONIC warm-start checkpoint is missing {name}")
    try:
        metadata = load_checkpoint_metadata(
            root,
            allow_git_commit_mismatch=allow_git_commit_mismatch,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Vast SONIC warm-start checkpoint is invalid: {exc}"
        ) from exc
    if not isinstance(metadata, SonicCheckpointMetadata):
        raise ValueError("Vast SONIC warm start requires a typed SONIC checkpoint")
    if metadata.dataset.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("SONIC warm-start checkpoint dataset fingerprint differs")
    if metadata.source_revision != selected_commit and not allow_git_commit_mismatch:
        raise ValueError("SONIC warm-start checkpoint source revision differs")
    release = load_sonic_release_artifact(release_artifact_path)
    if metadata.release_checkpoint_sha256.removeprefix(
        "sha256:"
    ) != release.checkpoint_sha256.removeprefix(
        "sha256:"
    ) or metadata.release_artifact_sha256.removeprefix(
        "sha256:"
    ) != release.artifact_sha256.removeprefix("sha256:"):
        raise ValueError("SONIC warm-start checkpoint release artifact differs")
    return root


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


def _twist2_warm_start_checkpoint(checkpoint_path: Path, *, dataset_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if any(path.is_symlink() for path in checkpoint_path.rglob("*")):
        raise ValueError("TWIST2 warm-start checkpoint must not contain symbolic links")
    metadata_path = checkpoint_path / "metadata.json"
    try:
        envelope = json.loads(metadata_path.read_text(encoding="utf-8"))
        index = json.loads(
            Path(dataset_path).expanduser().resolve().read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"TWIST2 warm-start input does not exist: {exc.filename}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            "TWIST2 warm-start metadata or dataset is invalid JSON"
        ) from exc
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("task") != "twist2":
        raise ValueError("Vast TWIST2 warm start requires a typed TWIST2 checkpoint")
    recorded_dataset = metadata.get("dataset")
    if not isinstance(recorded_dataset, dict):
        raise ValueError("TWIST2 warm-start checkpoint has no dataset contract")
    for key in (
        "schema_version",
        "robot_fingerprint",
        "gmr_revision",
        "joint_names",
        "body_names",
    ):
        if recorded_dataset.get(key) != index.get(key):
            raise ValueError(f"TWIST2 warm-start dataset differs for {key}")
    if recorded_dataset.get("dataset_fingerprint") == index.get("dataset_fingerprint"):
        raise ValueError("TWIST2 warm start requires a different dataset fingerprint")
    for name in ("state", "continuation"):
        if not (checkpoint_path / name).is_dir():
            raise ValueError(f"TWIST2 warm-start checkpoint is missing {name}/")
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
    if config.experiment in {"twist_teacher", "twist_student", "twist2"}:
        return config.experiment.replace("_", "-")
    if config.experiment == "sonic":
        return "sonic-transfer"
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
    offers.sort(key=lambda offer: _offer_sort_key(offer, config))
    if offers:
        return offers
    raise RuntimeError(
        "no Vast offers matched the configured hardware and price limits"
    )


def _offer_sort_key(
    offer: VastOffer, config: VastTrainConfig
) -> tuple[int, float, float]:
    country = offer.geolocation.rsplit(",", 1)[-1].strip().upper()
    try:
        location_rank = config.preferred_countries.index(country)
    except ValueError:
        location_rank = len(config.preferred_countries)
    return location_rank, -offer.dlperf_per_dphtotal, offer.dph_total


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
                runtype="ssh_direct",
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
                    "reconciliation, so another offer will not be rented. "
                    f"Vast cause: {safe_exception_message(exc)}"
                ) from exc
            on_offer_unavailable()
            print(
                f"Vast offer {offer.id} could not be rented "
                f"({type(exc).__name__}); trying the next match."
            )
    raise OffersUnavailableError(
        "all matching Vast offers became unavailable during rental"
    )


def _is_explicit_offer_unavailable(exc: BaseException) -> bool:
    return exception_http_status(exc) in {400, 404, 409, 410, 422}


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
        "set +x\n"
        "install -d -m 0700 -o root -g root /root/.ssh\n"
        "if [ -f /root/.ssh/authorized_keys ]; then\n"
        "  chown root:root /root/.ssh/authorized_keys\n"
        "  chmod 0600 /root/.ssh/authorized_keys\n"
        "fi\n"
        f"nohup python3 -c {shlex.quote(program)} </dev/null >/dev/null 2>&1 &"
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


class OffersUnavailableError(RuntimeError):
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
        for offer_number, offer in enumerate(offers, start=1):
            instance_id: int | None = None
            failure: BaseException | None = None
            retry_offer = False
            try:
                instance_id = create_available_instance(
                    self.api,
                    [offer],
                    self.config,
                    self.label,
                    on_before_create=lambda: write_creating_state(
                        self.state_path, self.label
                    ),
                    on_offer_unavailable=lambda: self.state_path.unlink(
                        missing_ok=True
                    ),
                )
                write_active_state(self.state_path, instance_id, self.label)
                print(f"Created Vast instance {instance_id}; teardown is armed.")
                try:
                    wait_for_instance(
                        self.api,
                        instance_id,
                        self.config.ready_timeout_seconds,
                    )
                except TimeoutError:
                    retry_offer = True
                    raise
                try:
                    ssh_target = wait_for_ssh(
                        self.api,
                        instance_id,
                        self.config.ssh_timeout_seconds,
                    )
                except TimeoutError:
                    retry_offer = True
                    raise
                if selection.dataset_files:
                    try:
                        speed = measure_upload_mib_per_second(
                            ssh_target,
                            size_mib=self.config.route_probe_mib,
                            timeout_seconds=self.config.ssh_timeout_seconds,
                        )
                    except (OSError, RuntimeError, subprocess.SubprocessError):
                        retry_offer = True
                        raise
                    print(f"Measured direct SSH upload: {speed:.1f} MiB/s.")
                    if speed < self.config.min_upload_mib_per_second:
                        retry_offer = True
                        raise RuntimeError(
                            f"offer {offer.id} upload route is too slow: "
                            f"{speed:.1f} MiB/s < "
                            f"{self.config.min_upload_mib_per_second:.1f} MiB/s"
                        )
                remote_runner(
                    ssh_target,
                    selection,
                    self.config,
                    run_name,
                    wandb_key=wandb_key,
                    github_token=github_token,
                    instance_api=self.api,
                    instance_id=instance_id,
                )
            except BaseException as exc:
                failure = exc
                retry_offer = retry_offer or isinstance(exc, OffersUnavailableError)
            finally:
                if instance_id is not None:
                    teardown_failure = self._destroy(instance_id)
                    if teardown_failure is not None:
                        failure = teardown_failure
                        retry_offer = False
            if failure is None:
                return
            if retry_offer and offer_number < len(offers):
                print(f"Rejected offer {offer.id}; trying the next nearby offer.")
                continue
            raise failure
        raise RuntimeError("all matching Vast offers were exhausted")

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
        raise RuntimeError(
            "dataset-backed Vast training requires rsync on the launch machine"
        )

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
