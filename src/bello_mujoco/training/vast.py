"""Vast.ai provisioning and remote training orchestration."""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import get_args
from urllib.parse import urlparse

import keyring

from bello_mujoco.configuration import (
    ExperimentName,
    is_motion_experiment,
    is_sonic_transfer_experiment,
    is_supermimic_experiment,
    profile_config_relative_path,
)
from bello_mujoco.training.vast_provider import (
    VastApi,
    VastOffer,
    VastSdkClient,
    exception_http_status,
)
from bello_mujoco.training.vast_remote import (
    RemoteTrainingFailed,
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
    min_gpu_ram_gb: int = 0
    disk_gb: int = 80
    min_reliability: float = 0.98
    min_cuda: float = 13.0
    image: str = "vastai/base-image:cuda-13.0.3-auto"
    search_limit: int = 50


@dataclass(frozen=True)
class VastTrainConfig:
    """Resolved configuration for one complete Vast training lifecycle."""

    experiment: ExperimentName = "bello_locomotion"
    motion_path: Path | None = None
    motion_dataset_artifact: str | None = None
    pretrained_revision: str | None = None
    evaluation_clips: tuple[str, ...] = ()
    evaluation_video_clips: tuple[str, ...] | None = None
    evaluation_workers: int = 1
    evaluate_training_split: bool = False
    total_timesteps: int | None = None
    num_envs: int | None = None
    learning_rate: float | None = None
    end_effector_payload_mass_kg: float | None = None
    max_price: float | None = None
    branch: str | None = None
    run_name: str | None = None
    skip_confirmation: bool = False
    dry_run: bool = False
    remote_dir: str = "/workspace/bello_mujoco"
    ready_timeout_seconds: int = 900
    ssh_timeout_seconds: int = 900
    preferred_countries: tuple[str, ...] = ("KR", "JP", "TW", "HK", "SG")
    job_timeout_seconds: int = 43_200
    progress_timeout_seconds: int = 1_800
    reconcile_retries: int = 3
    offer: VastOfferConfig = field(default_factory=VastOfferConfig)


@dataclass(frozen=True)
class GitSelection:
    branch: str
    commit: str
    config_path: PurePosixPath
    repo_url: str


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
    if config.experiment not in get_args(ExperimentName):
        raise ValueError("--experiment is invalid")
    if type(config.evaluation_workers) is not int or config.evaluation_workers <= 0:
        raise ValueError("evaluation_workers must be a positive integer")
    if not is_supermimic_experiment(config.experiment) and (
        config.evaluation_workers != 1
        or config.evaluation_video_clips is not None
        or config.evaluate_training_split
    ):
        raise ValueError(
            "evaluation_workers, evaluation_video_clips and evaluate_training_split require SuperMimic"
        )
    if config.evaluation_video_clips is not None and (
        len(set(config.evaluation_video_clips)) != len(config.evaluation_video_clips)
        or not set(config.evaluation_video_clips).issubset(config.evaluation_clips)
    ):
        raise ValueError(
            "evaluation_video_clips must be a unique subset of evaluation_clips"
        )
    if len(set(config.evaluation_clips)) != len(config.evaluation_clips) or any(
        not name or Path(name).name != name or Path(name).suffix
        for name in config.evaluation_clips
    ):
        raise ValueError("evaluation_clips must be unique clip stems")
    if is_motion_experiment(config.experiment):
        if config.motion_path is None:
            raise ValueError("--motion-path is required for the motion experiment")
        path = config.motion_path.expanduser()
        if path.suffix != ".npz" or not path.is_file():
            raise ValueError("motion_path must name an existing local .npz clip")
        with path.open("rb"):
            pass
        if config.end_effector_payload_mass_kg is not None and (
            not math.isfinite(config.end_effector_payload_mass_kg)
            or config.end_effector_payload_mass_kg < 0.0
        ):
            raise ValueError(
                "--end-effector-payload-mass-kg must be finite and nonnegative"
            )
    elif is_sonic_transfer_experiment(config.experiment) or is_supermimic_experiment(
        config.experiment
    ):
        if (
            not config.motion_dataset_artifact
            or re.fullmatch(
                r".+:(?:v[0-9]+|[0-9a-f]{32,})", config.motion_dataset_artifact
            )
            is None
        ):
            raise ValueError(
                "--motion-dataset-artifact must include an immutable version or digest"
            )
        if is_sonic_transfer_experiment(config.experiment) and (
            config.pretrained_revision is None
            or re.fullmatch(r"[0-9a-f]{40}", config.pretrained_revision) is None
        ):
            raise ValueError(
                "--pretrained-revision must be a 40-character lowercase commit"
            )
        if config.motion_path is not None:
            raise ValueError("dataset experiments do not accept --motion-path")
        if config.end_effector_payload_mass_kg is not None:
            raise ValueError("dataset experiments do not accept a payload mass")
        if (
            is_supermimic_experiment(config.experiment)
            and config.pretrained_revision is not None
        ):
            raise ValueError(
                "SuperMimic trains from scratch; omit --pretrained-revision"
            )
    elif (
        config.motion_path is not None
        or config.motion_dataset_artifact is not None
        or config.pretrained_revision is not None
        or config.evaluation_clips
        or config.end_effector_payload_mass_kg is not None
    ):
        raise ValueError(
            "motion and SONIC inputs are only valid for their matching experiments"
        )
    if config.total_timesteps is not None and config.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if config.num_envs is not None and config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.offer.num_gpus != 1:
        raise ValueError(
            "training currently supports exactly one GPU, not distributed PPO"
        )
    if (
        config.offer.min_gpu_ram_gb < 0
        or config.offer.min_cpu_ram_gb <= 0
        or config.offer.disk_gb <= 0
    ):
        raise ValueError("invalid offer RAM or disk constraint")
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
    if config.job_timeout_seconds <= 0 or config.progress_timeout_seconds <= 0:
        raise ValueError("job and progress timeouts must be positive")
    if config.progress_timeout_seconds > config.job_timeout_seconds:
        raise ValueError("progress-timeout-seconds must not exceed job-timeout-seconds")
    if config.reconcile_retries <= 0:
        raise ValueError("reconcile-retries must be positive")
    remote_path = PurePosixPath(config.remote_dir)
    if (
        not remote_path.is_absolute()
        or remote_path.anchor != "/"
        or PurePosixPath("/workspace") not in remote_path.parents
        or ".." in remote_path.parts
        or str(remote_path) != config.remote_dir
        or re.fullmatch(r"/[A-Za-z0-9._/-]+", config.remote_dir) is None
    ):
        raise ValueError("--remote-dir must be a normalized strict child of /workspace")


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
    submodule_paths = ["deps/bello-stack"]
    if is_sonic_transfer_experiment(config.experiment) or is_supermimic_experiment(
        config.experiment
    ):
        submodule_paths.append("deps/gear_sonic")
    for submodule_path in submodule_paths:
        submodule_entry = run_git(
            repo_root, "ls-tree", commit, "--", submodule_path
        ).split()
        if (
            len(submodule_entry) != 4
            or submodule_entry[:2] != ["160000", "commit"]
            or submodule_entry[3] != submodule_path
        ):
            raise ValueError(
                f"{submodule_path!r} is not a submodule on origin/{branch}"
            )
    repo_url = normalized_repo_url(run_git(repo_root, "remote", "get-url", "origin"))
    return GitSelection(
        branch=branch,
        commit=commit,
        config_path=config_path,
        repo_url=repo_url,
    )


def _task_name(config: VastTrainConfig) -> str:
    if is_supermimic_experiment(config.experiment):
        return "supermimic"
    if is_sonic_transfer_experiment(config.experiment):
        return "sonic-transfer"
    if is_motion_experiment(config.experiment):
        return "motion-tracking"
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
        f"gpu_ram>={offer.min_gpu_ram_gb}",
        f"disk_space>={offer.disk_gb}",
        f"cuda_max_good>={offer.min_cuda}",
        f"dph_total<={config.max_price}",
        # The SDK's duration query is in days; returned end_date is Unix time.
        f"duration>={watchdog_deadline(config, now=0) / 86400:.9f}",
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
        and offer.gpu_ram_mb >= desired.min_gpu_ram_gb * 1000
        and offer.cuda_max_good >= desired.min_cuda
        and offer.disk_space_gb >= desired.disk_gb
        and offer.direct_port_count >= 1
        and math.isfinite(offer.end_date)
        and offer.end_date >= watchdog_deadline(config)
    )


def print_offers(offers: Sequence[VastOffer], *, limit: int = 5) -> None:
    print("Matching Vast offers (best value first):")
    print(
        "  offer       $/hr   reliability  location             CPU RAM  GPU RAM  expires (UTC)"
    )
    for offer in offers[:limit]:
        print(
            f"  {offer.id:<10d}  {offer.dph_total:>5.3f}  "
            f"{offer.reliability:>10.3f}  {offer.geolocation[:20]:<20} "
            f"{offer.cpu_ram_mb / 1024:>6.1f}G  {offer.gpu_ram_mb / 1024:>6.1f}G  "
            f"{datetime.fromtimestamp(offer.end_date, UTC):%Y-%m-%d %H:%M}"
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
) -> int:
    """Rent the first still-available offer without exposing SDK error URLs."""

    for offer in offers:
        if not config.skip_confirmation:
            confirm_rental(offer)
        if not offer_matches(offer, config):
            print(
                f"Vast offer {offer.id} no longer meets the rental constraints; skipping."
            )
            continue
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
                    "reconciliation, so another offer will not be rented. "
                    f"Vast cause: {safe_exception_message(exc)}"
                ) from exc
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
        + config.job_timeout_seconds
        + WATCHDOG_TEARDOWN_GRACE_SECONDS
    )


def watchdog_onstart_script(config: VastTrainConfig) -> str:
    program = Path(__file__).with_name("vast_host.py").read_text()
    program += f"\nsupervise({watchdog_deadline(config)})\n"
    return (
        f"set +x\nnohup python3 -u -c {shlex.quote(program)} "
        "</dev/null >/tmp/bello-vast-supervisor.log 2>&1 &"
    )


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


class OffersUnavailableError(RuntimeError):
    pass


def ensure_instance_destroyed(
    api: VastApi,
    instance_id: int,
    label: str,
    *,
    grace_seconds: float = 45,
    timeout_seconds: float = 120,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Verify host cleanup after a confirmed job exit, with a local fallback."""
    started = clock()
    deadline = started + timeout_seconds
    next_delete = started + grace_seconds
    while True:
        try:
            info = api.show_instance(instance_id)
        except Exception as error:
            print(f"Vast cleanup status retry: {type(error).__name__}")
        else:
            if info is None:
                print(f"Vast instance {instance_id}: deletion verified.")
                return
            if int(info["id"]) != instance_id or info.get("label") != label:
                raise ValueError("Vast cleanup instance identity does not match")
            if clock() >= next_delete:
                print(f"Vast instance {instance_id}: requesting controller cleanup.")
                try:
                    api.destroy_instance(instance_id)
                except Exception as error:
                    print(f"Vast cleanup deletion retry: {type(error).__name__}")
                next_delete = clock() + 5
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(
                f"Vast instance {instance_id}: deletion could not be verified"
            ) from None
        sleep(min(5, remaining))


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
    if config.motion_path is not None:
        print(
            f"Selected local motion upload {config.motion_path.expanduser().resolve()}."
        )

    if config.dry_run:
        print(f"Offer query: {offer_query(config)}")
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
    offers = find_offers(api, config)
    print_offers(offers)
    label = f"bello-{run_name}"
    instance_id = create_available_instance(api, offers, config, label)
    print(
        f"Vast instance {instance_id}: host owns teardown; local exit leaves it running."
    )
    wait_for_instance(api, instance_id, config.ready_timeout_seconds)
    target = wait_for_ssh(api, instance_id, config.ssh_timeout_seconds)
    try:
        remote_runner(
            target,
            selection,
            config,
            run_name,
            wandb_key=wandb_key,
            github_token=github_token,
        )
    except RemoteTrainingFailed:
        ensure_instance_destroyed(api, instance_id, label)
        raise
    ensure_instance_destroyed(api, instance_id, label)
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
